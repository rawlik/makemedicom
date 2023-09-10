import os
import sys
import datetime
import argparse
import logging
from typing import Tuple
import importlib.metadata

import numpy as np
import h5py
import pydicom


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
    a: np.ndarray, dtype: np.dtype, dataspan=None
) -> Tuple[float, float, np.ndarray]:
    """
    The value b in the relationship between stored values (SV)
    in Pixel Data (7FE0,0010) and the output units specified in Rescale Type (0028,1054).

    Output units = m*SV + b.

    ref. https://dicom.innolitics.com/ciods/digital-x-ray-image/dx-image/00281052
    """
    if dataspan is None:
        datarange = np.max(a).astype(np.float64) - np.min(a).astype(np.float64)
        datarange *= 0.99
        datamin = np.min(a).astype(np.float64)
    else:
        datamin = dataspan[0]
        datarange = dataspan[1] - dataspan[0]

    dtyperange = float(np.iinfo(dtype).max) - float(np.iinfo(dtype).min)
    dtypemin = float(np.iinfo(dtype).min)

    slope = dtyperange / datarange
    offset = dtypemin - datamin * slope

    scaled = np.array(a * slope + offset).astype(dtype)

    # the inverse transform
    # a = (scaled - offset) / slope
    # a = scaled / slope - offset / slope
    invslope = 1 / slope
    invintercept = -offset / slope

    return invslope, invintercept, scaled


class Study:
    def __init__(
        self,
        description: str = "study",
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

    def set_in_dataset(self, ds: pydicom.Dataset):
        ds.StudyDescription = self.StudyDescription
        ds.StudyInstanceUID = self.StudyInstanceUID
        ds.StudyID = self.StudyID
        ds.StudyDate = self.StudyDate
        ds.StudyTime = self.StudyTime


def create_dicom_dataset():
    # endianess = {
    #     ">": "big",
    #     "<": "little",
    #     "=": sys.byteorder,
    #     "|": "not applicable",
    # }[np.dtype(dtype).byteorder]
    endianess = "little"

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
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesNumber = 1

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


def image_to_dicom(
    d: np.ndarray, dtype: np.dtype, filename: str, study: Study = None
) -> pydicom.Dataset:
    ds = create_dicom_dataset()

    # TODO make radiography
    make_dataset_CT(ds)

    if study is not None:
        study.set_in_dataset(ds)

    slope, intercept, scaled = normalise_for_dicom(d, dtype=dtype)

    set_pixel_data_from_array(ds, scaled)

    # ds.RescaleSlope = f"{slope:f}"[:16]
    # ds.RescaleIntercept = f"{intercept:f}"[:16]

    validate_dataset(ds)
    ds.save_as(filename)

    return ds


def make_dataset_CT(ds: pydicom.Dataset, voxelsize=1):
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.ImageType = "ORIGINAL\\PRIMARY\\AXIAL"
    ds.Modality = "CT"

    ds.PixelSpacing = [voxelsize, voxelsize]
    ds.SliceThickness = voxelsize
    ds.SpacingBetweenSlices = voxelsize


def volume_to_dicom(
    d: np.ndarray,
    dtype: np.dtype,
    folder: str,
    voxelsize: float = 1,
    study: Study = None,
) -> pydicom.Dataset:
    if study is None:
        study = Study()

    seriesInstanceUID = pydicom.uid.generate_uid()

    dataspan = d.min(), d.max()

    for i in range(d.shape[0]):
        ds = create_dicom_dataset()

        make_dataset_CT(ds, voxelsize=voxelsize)

        # all datasets are part of the same study and series
        study.set_in_dataset(ds)
        ds.SeriesInstanceUID = seriesInstanceUID
        ds.InstanceNumber = i

        slope, intercept, scaled = normalise_for_dicom(
            d[i, ...], dtype=dtype, dataspan=dataspan
        )

        set_pixel_data_from_array(ds, scaled)

        # ds.RescaleSlope = f"{slope:f}"[:16]
        # ds.RescaleIntercept = f"{intercept:f}"[:16]

        validate_dataset(ds)
        ds.save_as(os.path.join(folder, f"{i:08d}.dcm"))


def entrypoint():
    parser = argparse.ArgumentParser(
        prog="makemedicom",
        description="Converts image files to DICOM. At the moment the following input formats are supported: hdf5.",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "file",
        type=str,
        nargs="+",
        help="The files to read.",
    )

    parser.add_argument(
        "-s",
        "--study",
        type=str,
        help="Treat all datasets as a part of a single study with this id.",
    )

    parser.add_argument(
        "--group2study",
        action="store_true",
        help="Use hdf5 group names as study.",
    )

    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    dtype = np.int16

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)

    if args.study is not None:
        study = Study(studyid=args.study)
    else:
        study = None

    for filename in args.file:
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        if "." in basename:
            fbasename, _, extension = basename.rpartition(".")
        else:
            logging.error("The file has no extension.")
            exit

        if extension in ["h5", "hdf5", "n5"]:

            def visit_hdf5_object(name, d):
                if isinstance(d, h5py.Dataset):
                    logging.debug(f"Found dataset: {name} {d.shape} {d.dtype}")
                    outpath = os.path.join(dirname, fbasename, name)

                    if args.group2study and (args.study is None):
                        study = Study(studyid=name)

                    if len(d.shape) == 2:
                        logging.info(f"Writing image {filename}/{name}")
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        image_to_dicom(d, dtype, outpath + ".dcm", study=study)
                    elif len(d.shape) == 3:
                        logging.info(f"Writing volume {filename}/{name}")
                        os.makedirs(outpath, exist_ok=True)
                        # we need to read the whole array to know the
                        # minimum and maximum values
                        volume_to_dicom(d[...], dtype, outpath, study=study)

            with h5py.File(filename, "r") as file:
                file.visititems(visit_hdf5_object)
        else:
            logging.error(f"File format {extension} not implemented.")


if __name__ == "__main__":
    entrypoint()
