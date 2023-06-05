import warnings
from typing import Optional, Tuple, Any, List

import eagerpy as ep
import numpy as np
import tensorflow as tf
from PIL import Image
from foolbox.models import Model
from foolbox.types import Bounds


"""
Based on foolbox.utils.samples:
1. if need get use defined images, set 'paths' a list of image file path
   NOTE : when paths be set, this api only return 'images', no 'labels'

2. if 'paths' is none, you can use this api as the same as foolbox.utils.samples
"""


def samples(
        fmodel: Model,
        kmodel: Any,
        dataset: str = "imagenet",
        index: int = 0,
        batchsize: int = 1,
        shape: Tuple[int, int] = (224, 224),
        data_format: Optional[str] = "channels_last",
        bounds: Tuple[float, float] = None,
        paths=None,
        user_define_labels=None
) -> Any:
    """
    Based on foolbox.utils.samples.
    """
    # use this api as the same as foolbox.utils.samples if paths is None
    if paths is None and user_define_labels is None:
        import foolbox as fb
        if hasattr(fmodel, "data_format"):
            if data_format is None:
                data_format = fmodel.data_format  # type: ignore
            elif data_format != fmodel.data_format:  # type: ignore
                raise ValueError(
                    f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"
                    # type: ignore
                )
        elif data_format is None:
            raise ValueError(
                "data_format could not be inferred, please specify it explicitly"
            )

        if bounds is None:
            bounds = fmodel.bounds

        images, labels = fb.utils.samples(
            fmodel=fmodel,
            dataset=dataset,
            index=index,
            batchsize=batchsize,
            shape=shape,
            data_format=data_format,
            bounds=Bounds(*bounds)
        )
        return images, labels

    # fmodel is a instance of ApplicationModel : amodel
    # images get from local files
    # return style the same as foolbox.utils.samples()
    if fmodel is not None and paths is not None:
        # labels get from amodel
        if user_define_labels is None:
            warnings.warn(f"Not implemented.")
            return None, None
        # labels get from use_define_labels
        else:
            images, unuseful_labels = _samples(
                dataset=dataset,
                index=index,
                batchsize=batchsize,
                shape=shape,
                data_format=data_format,
                bounds=Bounds(*bounds),
                paths=paths,
                kmodel=None
            )
            labels = np.asarray(user_define_labels)
            labels = tf.convert_to_tensor(labels, tf.int64)
            labels = ep.astensor(labels)  # convert type into <class 'eagerpy.tensor.tensorflow.TensorFlowTensor'>
            images = tf.convert_to_tensor(images, tf.float32)
            return images, labels

    # the last situation
    # kmodel is a Tensorflow style model
    # images get from local files
    # labels get from kmodel
    if kmodel is None:
        raise ValueError(
            "kmodel must not be None when fmodel is None and paths in not None"
        )
    images, labels = _samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        shape=shape,
        data_format=data_format,
        bounds=Bounds(*bounds),
        paths=paths,
        kmodel=kmodel
    )

    labels = tf.convert_to_tensor(labels, tf.int64)
    labels = ep.astensor(labels)  # convert type into <class 'eagerpy.tensor.tensorflow.TensorFlowTensor'>
    images = tf.convert_to_tensor(images, tf.float32)

    return images, labels


def _samples(
        dataset: str,
        index: int,
        batchsize: int,
        shape: Tuple[int, int],
        data_format: str,
        bounds: Bounds,
        paths: List,
        kmodel: Any
) -> Tuple[Any, Any]:
    """
    Based on foolbox.utils._samples.
    """

    # from PIL import Image

    images = []
    labels = []

    if index + batchsize > len(paths):
        warnings.warn(
            "index + batchsize must <= len(path)"
        )
        exit(1)

    for idx in range(index, index + batchsize):  # get filename and label
        file = paths[idx]

        # open file
        image = Image.open(file)

        if dataset == "imagenet":
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3

        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        if kmodel is not None:
            label = np.argmax(kmodel.predict(image[np.newaxis, :, :, ::-1]))
        else:
            label = -1  # this is unuseful label

        images.append(image)
        labels.append(label)

    images_ = np.stack(images)
    labels_ = np.array(labels)

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images_, labels_


