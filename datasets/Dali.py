from nvidia.dali import pipeline_def
from nvidia.dali.fn.readers import file
from nvidia.dali.fn.decoders import image, image_random_crop
from nvidia.dali.fn import resize
from typing import Optional
from nvidia.dali.fn import crop_mirror_normalize, cast
from nvidia.dali.fn.random import coin_flip
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.types as types
@pipeline_def
def get_pipeline(
		file_root: str,
		random_shuffle: bool = True,
		training: bool = True,
		size: int = 224,
		validation_size: Optional[int] = 256,  # though it is not used, but is necessary to define the pipeline
		decoder_device: str = 'mixed',
		device: str = 'gpu',
):
	images, labels = file(
		file_root=file_root,
		random_shuffle=random_shuffle,
		# The name could be anything,
		# 'Reader' was picked at random
		name='Reader',
	)

	if training:
		images = image_random_crop(
			images,
			random_area=[0.08, 1.0],
			random_aspect_ratio=[0.75, 1.3],
			device=decoder_device,
		)

		images = resize(
			images,
			size=size,
			device=device,
		)

		mirror = coin_flip(
			probability=0.5,
		)

	else:
		images = image(
			images,
			device=decoder_device,
		)

		images = resize(
			images,
			size=size,
			mode='not_smaller',
			device=device,
		)

		mirror = False

	images = crop_mirror_normalize(
		images,
		crop=(size, size),
		mirror=mirror,
		mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
		std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
		device=device,
		dtype=types.FLOAT,
	)

	if device == 'gpu':
		labels = labels.gpu()
		# dali in default return label as int32 but torch requires int64
		labels = cast(labels, dtype=types.INT64)
	return images, labels


def get_dalli_loader(args, training=True):
	if training:
		training_pipeline = get_pipeline(
			# This would be the batch size one desires for their data loader
			batch_size=args.batch_size,
			num_threads=args.num_workers,
			# GPU IDs start at 0
			device_id=0,
			file_root=args.dataset_path+'/train',
			random_shuffle=True,
			training=True,
			size=224,
			decoder_device='mixed',
			device='gpu',
		)
		training_pipeline.build()
		training_dataloader = DALIClassificationIterator(
			pipelines=training_pipeline,
			reader_name='Reader',
			# last_batch_policy tells DALI how to treat the last
			# batch if it is not complete, that is, there are not
			# enough items left to fill an entire batch
			# LastBatchPolicy.PARTIAL means let the last batch be not full
			# For instance, if the data is [1, 2, 3] and the batch size is 2,
			# the first batch would be [1, 2] and the second would be [3]
			last_batch_policy=LastBatchPolicy.PARTIAL,
			# After DALI iterates through the dataset (i.e., after each epoch),
			# the data loader must be reset, which can be done via dataloader.reset()
			# Alternatively, auto_reset can be set to True so the loader automatically resets
			# after the entire dataset has been iterated through
			auto_reset=True,
		)
		return training_dataloader
	else:
		validation_pipeline = get_pipeline(
			batch_size=args.batch_size,
			num_threads=args.num_workers,
			device_id=0,
			file_root=args.dataset_path + '/val',
			random_shuffle=False,
			training=False,
			size=224,
			validation_size=256,
			decoder_device='mixed',
			device='gpu',
		)
		validation_pipeline.build()
		validation_dataloader = DALIClassificationIterator(
			pipelines=validation_pipeline,
			reader_name='Reader',
			last_batch_policy=LastBatchPolicy.PARTIAL,
			auto_reset=True,
		)
		return validation_dataloader


def get_dali_loaders(args):
	train_dataloader = get_dalli_loader(args, training=True)
	val_dataloader = get_dalli_loader(args, training=False)
	return train_dataloader, val_dataloader
