import os
import sys
import datetime
import argparse
import logging
from typing import Tuple, List
import importlib.metadata

import numpy as np
import h5py
import pydicom
import pydicom.fileset
from pydicom.pixel_data_handlers.util import apply_modality_lut


try:
    __version__ = importlib.metadata.version("makemedicom")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"


def hello():
    print("hello!")
    print(h5py.__version__)
    print(pydicom.__version__)
    print("hello")


def normalise_for_dicom(
    a: np.ndarray,
    dtype: np.dtype,
    dataspan: Tuple[float, float] = None,
    ds: pydicom.Dataset = None,
) -> Tuple[float, float, np.ndarray]:
    """
    The value b in the relationship between stored values (SV)
    in Pixel Data (7FE0,0010) and the output units specified in Rescale Type (0028,1054).

    Output units = m*SV + b.

    If the dataset ds is given, set the RescaleSlope and RescaleIntercept attributes.

    ref. https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281052
    """
    if np.issubdtype(dtype, np.integer):
        if dataspan is None:
            datamin = np.min(a).astype(np.float64)
            datamax = np.max(a).astype(np.float64)
        else:
            datamin = dataspan[0]
            datamax = dataspan[1]

        logging.debug(f"datamin {datamin}, datamax {datamax}")
        datarange = datamax - datamin
        datarange *= 0.999

        dtyperange = float(np.iinfo(dtype).max) - float(np.iinfo(dtype).min)
        dtypemin = float(np.iinfo(dtype).min)

        slope = dtyperange / datarange
        logging.debug(f"dtyperange {dtyperange}, datarange {datarange}, slope {slope}")
        offset = dtypemin - datamin * slope

        scaled = np.array(a * slope + offset, dtype=dtype)

        # the inverse transform
        # a = (scaled - offset) / slope
        # a = scaled / slope - offset / slope
        invslope = 1 / slope
        invintercept = -offset / slope

        if ds is not None:
            ds.RescaleSlope = f"{invslope:e}"
            ds.RescaleIntercept = f"{invintercept:e}"

        return invslope, invintercept, scaled
    else:
        return 1, 0, np.array(a).astype(dtype)

class Series:
    def __init__(
        self,
        description: str = "",
        seriesnumber: str = "1",
    ) -> None:
        self.SeriesDescription = description
        self.SeriesInstanceUID = pydicom.uid.generate_uid()
        self.SeriesNumber = seriesnumber

    def set_in_dataset(self, ds: pydicom.Dataset):
        ds.SeriesDescription = self.SeriesDescription
        ds.SeriesInstanceUID = self.SeriesInstanceUID
        ds.SeriesNumber = self.SeriesNumber


class Study:
    def __init__(
        self,
        description: str = "",
        studyid: str = "1",
        dt: datetime.datetime = None,
    ) -> None:
        self.StudyDescription = description
        self.StudyInstanceUID = pydicom.uid.generate_uid()
        self.StudyID = studyid

        if dt is None:
            dt = datetime.datetime.now()

        self.StudyDate = dt.strftime("%Y%m%d")
        self.StudyTime = dt.strftime("%H%M%S.%f")

        self.series = []

    def set_in_dataset(self, ds: pydicom.Dataset):
        ds.StudyDescription = self.StudyDescription
        ds.StudyInstanceUID = self.StudyInstanceUID
        ds.StudyID = self.StudyID
        ds.StudyDate = self.StudyDate
        ds.StudyTime = self.StudyTime

    def add_series(self, description: str = "") -> Series:
        number = len(self.series) + 1
        series = Series(description=description, seriesnumber=number)
        self.series.append(series)

        return series


def create_dicom_dataset(ds: pydicom.dataset.Dataset = None):
    # endianess = {
    #     ">": "big",
    #     "<": "little",
    #     "=": sys.byteorder,
    #     "|": "not applicable",
    # }[np.dtype(dtype).byteorder]
    endianess = "little"

    if ds is None:
        ds = pydicom.dataset.Dataset()
    ds.ensure_file_meta()

    # set the preamble from a valid DICOM file
    preamble = pydicom.dcmread(pydicom.data.get_testdata_file("CT_small.dcm")).preamble
    ds.preamble = preamble

    # Set the transfer syntax
    ds.is_little_endian = endianess == "little"
    ds.is_implicit_VR = True
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # encoding
    ds.SpecificCharacterSet = "ISO_IR 100"

    # the source application is makemedicom
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.ImplementationVersionName = "v" + __version__
    ds.file_meta.SourceApplicationEntityTitle = "makemedicom"

    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID

    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    # default patient
    ds.PatientName = "Anonymous Patient"
    ds.PatientID = "123456"

    # default Study
    study = Study()
    study.set_in_dataset(ds)

    # series
    series = study.add_series()
    series.set_in_dataset(ds)

    ds.AcquisitionNumber = 1
    ds.InstanceNumber = 1

    # default interpretation
    ds.PhotometricInterpretation = "MONOCHROME2"

    # default image type
    ds.ImageType = "ORIGINAL\\PRIMARY"

    # default pixel values
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 1
    ds.PixelPaddingValue = 0

    return ds


def validate_dataset(ds: pydicom.Dataset):
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)


def set_pixel_data_from_array(ds: pydicom.Dataset, d: np.ndarray, endianess="little"):
    ds.Rows = d.shape[-2]
    ds.Columns = d.shape[-1]
    bitsize = d.dtype.itemsize * 8
    ds.BitsStored = bitsize
    ds.BitsAllocated = bitsize
    ds.HighBit = (bitsize - 1) if endianess == "little" else 0
    ds.PixelRepresentation = 1
    ds.PixelData = d.tobytes()


def dicom_to_dicom(
    ds: pydicom.dataset.Dataset,
    filename: str,
    dtype: str = None,
    study: Study = None,
    series: Series = None,
    fileset: pydicom.fileset.FileSet = None,
) -> pydicom.Dataset:
    if dtype is not None:
        d = ds.pixel_array
        slope, intercept, scaled = normalise_for_dicom(
            apply_modality_lut(d, ds), dtype=dtype
        )
        ds = create_dicom_dataset(ds)
        set_pixel_data_from_array(ds, scaled)
        ds.RescaleSlope = f"{slope:f}"[:16]
        ds.RescaleIntercept = f"{intercept:f}"[:16]
    else:
        ds = create_dicom_dataset(ds)

    make_dataset_DigitalXRayImageStorageForPresentation(ds)

    if study is not None:
        study.set_in_dataset(ds)

    if series is not None:
        series.set_in_dataset(ds)

    rescaled = apply_modality_lut(ds.pixel_array, ds)
    ds.WindowCenter = rescaled.mean()
    ds.WindowWidth = rescaled.max() - rescaled.min()

    validate_dataset(ds)

    if fileset is None:
        ds.save_as(filename)
    else:
        fileset.add(ds)

    return ds


def image_to_dicom(
    d: np.ndarray,
    dtype: np.dtype,
    filename: str,
    study: Study = None,
    series: Series = None,
    fileset: pydicom.fileset.FileSet = None,
    **attrs,
) -> pydicom.Dataset:
    ds = create_dicom_dataset()

    make_dataset_DigitalXRayImageStorageForPresentation(ds)

    if study is not None:
        study.set_in_dataset(ds)

    if series is not None:
        series.set_in_dataset(ds)

    slope, intercept, scaled = normalise_for_dicom(d, dtype=dtype, ds=ds)

    set_pixel_data_from_array(ds, scaled)

    rescaled = apply_modality_lut(ds.pixel_array, ds)
    ds.WindowCenter = rescaled.mean()
    ds.WindowWidth = rescaled.max() - rescaled.min()

    for k in attrs:
        setattr(ds, k, attrs[k])

    validate_dataset(ds)

    if fileset is None:
        ds.save_as(filename)
    else:
        fileset.add(ds)

    return ds


def make_dataset_CT(ds: pydicom.Dataset, voxelsize=1):
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.ImageType = "ORIGINAL\\PRIMARY\\AXIAL"
    ds.Modality = "CT"

    ds.PixelSpacing = [voxelsize, voxelsize]
    ds.SliceThickness = voxelsize
    ds.SpacingBetweenSlices = voxelsize


def make_dataset_DigitalXRayImageStorageForPresentation(
    ds: pydicom.Dataset, pixelsize=1
):
    ds.file_meta.MediaStorageSOPClassUID = (
        pydicom.uid.DigitalXRayImageStorageForPresentation
    )
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.Modality = "DX"

    ds.PixelSpacing = [pixelsize, pixelsize]


def volume_to_dicom(
    d: np.ndarray,
    dtype: np.dtype,
    folder: str,
    voxelsize: float = 1,
    study: Study = None,
    series: Series = None,
    fileset: pydicom.fileset.FileSet = None,
    **attrs,
) -> List[pydicom.Dataset]:
    if study is None:
        study = Study()

    if series is None:
        series = study.add_series()

    datamean = d.mean()
    datamin = d.min()
    datamax = d.max()
    dataspan = datamin, datamax

    datasets = []

    for i in range(d.shape[0]):
        ds = create_dicom_dataset()

        make_dataset_CT(ds, voxelsize=voxelsize)

        # all datasets are part of the same study and series
        study.set_in_dataset(ds)
        series.set_in_dataset(ds)
        ds.InstanceNumber = i

        slope, intercept, scaled = normalise_for_dicom(
            d[i, ...], dtype=dtype, dataspan=dataspan, ds=ds
        )

        set_pixel_data_from_array(ds, scaled)

        ds.WindowCenter = datamean
        ds.WindowWidth = datamax - datamin

        for k in attrs:
            setattr(ds, k, attrs[k])

        validate_dataset(ds)

        if fileset is None:
            ds.save_as(os.path.join(folder, f"{i:08d}.dcm"))
        else:
            fileset.add(ds)

        datasets.append(ds)

    return datasets


def process_files(
    files,
    outpath,
    dtype: np.dtype,
    file2study: bool = False,
    studydescription: str = None,
    create_fileset: bool = False,
):
    if not file2study:
        study = Study()

        if studydescription is not None:
            study.StudyDescription = studydescription

    if create_fileset:
        fileset = pydicom.fileset.FileSet()
    else:
        fileset = None

    for filename in files:
        logging.info(f"Processing file {filename}")
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)

        if file2study:
            study = Study(description=basename)

        if "." in basename:
            fbasename, _, extension = basename.rpartition(".")
        else:
            logging.error(f"The file has no extension: {filename}")
            exit(1)

        fileoutpath = os.path.join(outpath, fbasename)

        if not create_fileset:
            os.makedirs(fileoutpath, exist_ok=True)

        if extension in ["h5", "hdf5", "n5"]:

            def visit_hdf5_object(name, d):
                if isinstance(d, h5py.Dataset):
                    logging.debug(f"Found dataset: {name} {d.shape} {d.dtype}")
                    objectoutpath = os.path.join(fileoutpath, name)

                    series = study.add_series(description=basename)
                    series.SeriesDescription += "/"
                    series.SeriesDescription += name

                    if len(d.shape) == 2:
                        logging.info(f"Writing image {filename}/{name}")
                        image_to_dicom(
                            d,
                            dtype,
                            objectoutpath + ".dcm",
                            series=series,
                            study=study,
                            fileset=fileset,
                        )

                    elif len(d.shape) == 3:
                        logging.info(f"Writing volume {filename}/{name}")
                        if not create_fileset:
                            os.makedirs(objectoutpath, exist_ok=True)
                        # we need to read the whole array to know the
                        # minimum and maximum values
                        volume_to_dicom(
                            d[...],
                            dtype,
                            objectoutpath,
                            series=series,
                            study=study,
                            fileset=fileset,
                        )

            with h5py.File(filename, "r") as file:
                file.visititems(visit_hdf5_object)
        elif extension in ["dcm"]:
            ds = pydicom.dcmread(filename)

            series = study.add_series(description=os.path.basename(filename))

            dicom_to_dicom(
                ds=ds,
                filename=fileoutpath + ".dcm",
                study=study,
                series=series,
                fileset=fileset,
                dtype=dtype,
            )

            if fileset is not None:
                fileset.add(ds)
        else:
            logging.error(f"File format {extension} not implemented.")

    if create_fileset:
        filesetpath = outpath
        logging.info(f"Writing fileset {filesetpath}")
        fileset.write(filesetpath)


def entrypoint():
    parser = argparse.ArgumentParser(
        prog="makemedicom",
        description="Converts image files to DICOM. At the moment the following input formats are supported: hdf5.",
    )

    parser.add_argument(
        "file",
        type=str,
        nargs="+",
        help="The files to read.",
    )

    parser.add_argument(
        "-o",
        "--outpath",
        required=True,
        type=str,
        help="The output path.",
    )

    parser.add_argument(
        "--file2study",
        type=bool,
        default=False,
        help="Create one study per file. By default create a single study for all files.",
    )

    parser.add_argument(
        "--studydescription",
        type=str,
        help="Description of the study.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="<i2",
        help="Convert the data to the given numpy dtype. Defaults to <i2.",
    )

    parser.add_argument(
        "--fileset",
        action="store_true",
        help="Save the files as a fileset creating a DICOMDIR file.",
    )

    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)

    process_files(
        files=args.file,
        outpath=args.outpath,
        dtype=np.dtype(args.dtype),
        file2study=args.file2study,
        studydescription=args.studydescription,
        create_fileset=args.fileset,
    )


if __name__ == "__main__":
    entrypoint()
