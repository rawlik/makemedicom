import os
import sys
import argparse
import logging
from typing import Tuple

import numpy as np
import h5py
import pydicom


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

    logging.debug(f"datamin {datamin}, dtypemin {dtypemin}")

    slope = dtyperange / datarange
    offset = dtypemin - datamin * slope

    scaled = np.array(a * slope + offset).astype(dtype)
    logging.debug(f"dtypemin {np.iinfo(dtype).min}, dtypemin {np.iinfo(dtype).max}")
    logging.debug(f"datamin {scaled.min()}, datamax {scaled.max()}")
    logging.debug(f"data dtype {scaled.dtype}")

    # the inverse transform
    # a = (scaled - offset) / slope
    # a = scaled / slope - offset / slope
    invslope = 1 / slope
    invintercept = -offset / slope

    return invslope, invintercept, scaled


def image_to_dicom(d: np.ndarray, dtype: np.dtype, filename: str) -> pydicom.Dataset:
    # ds = pydicom.dataset.FileDataset(
    #     filename, {}, file_meta=file_meta, preamble=b"\0" * 128
    # )
    # ds = pydicom.dcmread(pydicom.data.get_testdata_file("CT_small.dcm"))
    ds = pydicom.dataset.Dataset()

    ds.ensure_file_meta()
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.ImplementationVersionName = "v0.0.1"
    ds.file_meta.SourceApplicationEntityTitle = "makemedicom"
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.ImageType = "ORIGINAL\\PRIMARY\\AXIAL"
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"

    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesNumber = 1
    ds.AcquisitionNumber = 1
    ds.InstanceNumber = 1

    slope, intercept, scaled = normalise_for_dicom(d, dtype=dtype)

    endianess = {
        ">": "big",
        "<": "little",
        "=": sys.byteorder,
        "|": "not applicable",
    }[np.dtype(dtype).byteorder]

    # Set the transfer syntax
    ds.is_little_endian = endianess == "little"
    ds.is_implicit_VR = True

    ds.Rows = scaled.shape[-2]
    ds.Columns = scaled.shape[-1]
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    bitsize = scaled.dtype.itemsize * 8
    ds.BitsStored = bitsize
    ds.BitsAllocated = bitsize
    ds.HighBit = (bitsize - 1) if endianess == "little" else 0
    ds.PixelRepresentation = 1
    ds.PixelSpacing = "0.1\\0.1"
    ds.PixelData = scaled.tobytes()
    logging.debug(f"{slope:f}"[:16])
    ds.RescaleSlope = f"{slope:f}"[:16]
    ds.RescaleIntercept = f"{intercept:f}"[:16]
    # logging.debug(f"RescaleSlope: {ds.RescaleSlope}")
    # logging.debug(f"RescaleIntercept: {ds.RescaleIntercept}")

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "Anonymous^Patient"
    ds.PatientID = "123456"

    ds.save_as(filename)

    return ds


def volume_to_dicom(d: np.ndarray, dtype: np.dtype, folder: str) -> pydicom.Dataset:
    studyInstanceUID = pydicom.uid.generate_uid()
    seriesInstanceUID = pydicom.uid.generate_uid()

    dataspan = d.min(), d.max()

    for i in range(d.shape[0]):
        if i > 10:
            break

        # ds = pydicom.dataset.FileDataset(
        #     filename, {}, file_meta=file_meta, preamble=b"\0" * 128
        # )
        # ds = pydicom.dcmread(pydicom.data.get_testdata_file("CT_small.dcm"))
        ds = pydicom.dataset.Dataset()

        ds.ensure_file_meta()
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        ds.file_meta.ImplementationVersionName = "v0.0.1"
        ds.file_meta.SourceApplicationEntityTitle = "makemedicom"
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = "ORIGINAL\\PRIMARY\\AXIAL"
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"

        ds.StudyInstanceUID = studyInstanceUID
        ds.SeriesInstanceUID = seriesInstanceUID
        ds.SeriesNumber = 1
        ds.AcquisitionNumber = 1
        ds.InstanceNumber = i

        slope, intercept, scaled = normalise_for_dicom(
            d[i, ...], dtype=dtype, dataspan=dataspan
        )

        endianess = {
            ">": "big",
            "<": "little",
            "=": sys.byteorder,
            "|": "not applicable",
        }[np.dtype(dtype).byteorder]

        # Set the transfer syntax
        ds.is_little_endian = endianess == "little"
        ds.is_implicit_VR = True

        ds.Rows = scaled.shape[-2]
        ds.Columns = scaled.shape[-1]
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        bitsize = scaled.dtype.itemsize * 8
        ds.BitsStored = bitsize
        ds.BitsAllocated = bitsize
        ds.HighBit = (bitsize - 1) if endianess == "little" else 0
        ds.PixelRepresentation = 1
        ds.PixelSpacing = "0.1\\0.1"
        ds.PixelData = scaled.tobytes()
        logging.debug(f"{slope:f}"[:16])
        ds.RescaleSlope = f"{slope:f}"[:16]
        ds.RescaleIntercept = f"{intercept:f}"[:16]
        # logging.debug(f"RescaleSlope: {ds.RescaleSlope}")
        # logging.debug(f"RescaleIntercept: {ds.RescaleIntercept}")

        # Add the data elements -- not trying to set all required here. Check DICOM
        # standard
        ds.PatientName = "Anonymous^Patient"
        ds.PatientID = "123456"

        ds.save_as(os.path.join(folder, f"{i:08d}.dcm"))


if __name__ == "__main__":
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

    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()

    dtype = np.int16

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

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
                    if len(d.shape) == 2:
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        array_to_dicom(d, dtype, outpath + ".dcm")
                    elif len(d.shape) == 3:
                        os.makedirs(outpath, exist_ok=True)
                        # we need to read the whole array to know the
                        # minimum and maximum values
                        volume_to_dicom(d[...], dtype, outpath)

            with h5py.File(filename, "r") as file:
                file.visititems(visit_hdf5_object)
