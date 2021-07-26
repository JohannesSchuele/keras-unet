import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# TODO: Make masks on the fly from database labels
# TODO: Make interface to handle loading for inference


class EndoDataset(object):
    """
    Class to handle image data
    """
    def __init__(self,
                 num_classes,
                 image_directory=r'./endo_images',
                 labeled_image_subdirectory='.',  # TODO: Read this from database
                 masks_in_separate_dir=False,
                 mask_image_subdirectory='SegmentationClassPNG',  # TODO: Read this from database
                 image_size=(512, 512),
                 crop_pad_size=(1024, 1024),
                 one_hot_encode=True
                 ):
        """

        :param sql_database: path to sql lite database that has all the image names and info
        :param labeled_image_subdirectory: this is the folder that the images to use are located in. It should be at
                                           the same level as the database (in same directory)
        :param masks_in_separate_dir: Whether or not the masks are stored in separate files in a different folder.
                                      If True, the mask_image_subdirectory is used to locate the masks.
                                      If False, the masks are assumed to be stored in the alpha channel of the images.
        :param mask_image_subdirectory: this is the folder that the masks to use are located in. It should be at
                                        the same level as the database (in same directory) and the masks should have the
                                        same name as their corresponding image
        :param sql_image_tablename: the table name in the sql database that all the image data is in
        :param image_size: image size to crop or pad all images to
        """
        super(EndoDataset, self).__init__()
        self.one_hot = one_hot_encode
        self.labeled_image_subdirectory = labeled_image_subdirectory
        self.mask_image_subdirectory = mask_image_subdirectory
        self.crop_pad_size = tf.constant(crop_pad_size, dtype=tf.int32)
        self.image_size = tf.constant(image_size, dtype=tf.int32)
        self.data_directory = tf.constant(image_directory, dtype=tf.string)
        # self.sql_dataset = tf.data.experimental.SqlDataset('sqlite', sql_database,
        #                                                    f"SELECT * FROM {sql_image_tablename}",
        #                                                    (tf.int32, tf.string, tf.string, tf.int32))
        # self.dataset_filenames = self.sql_dataset.map(lambda *x: x[2])
        self.dataset_filenames = tf.data.Dataset.list_files(image_directory + '/' + self.labeled_image_subdirectory + '/*.jpg', shuffle=False)
        self._count_constant = tf.constant(0, dtype=tf.int64)
        self.image_modification_random = tf.random.uniform([])
        self.masks_in_separate_directory = masks_in_separate_dir
        self.num_classes = tf.constant(num_classes, dtype=tf.int32)
        # TODO: Read number of classes from database and be able to return it here.

    @tf.function
    def parse_image_mask_pair(self, img_path: tf.string) -> dict:
        """
        Load an image and its annotation (mask) and returning a dictionary.
        :param img_path: Image (not the mask) location. currently expects jpg
        :return: dictionary {'image': image tensor, 'segmentation_mask': mask tensor}
        """

        lbl_folder = tf.strings.join([self.data_directory, '/',  self.labeled_image_subdirectory, '/'])
        img_name = tf.strings.regex_replace(img_path, pattern=r'.*'+self.labeled_image_subdirectory +'/?(.*)', rewrite=r'\1')
        img_path = tf.strings.join([lbl_folder, img_name])
        img_path = tf.strings.regex_replace(img_path, "png", "jpg")
        tf_image = tf.io.read_file(img_path)
        tf_image = tf.image.decode_jpeg(tf_image, channels=3)
        tf_image = tf.image.convert_image_dtype(tf_image, tf.uint8)

        if self.masks_in_separate_directory:
            mask_path = tf.strings.regex_replace(img_path,
                                                 self.labeled_image_subdirectory,
                                                 self.mask_image_subdirectory)
            mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=0)
        else:
            mask = tf_image

        mask = tf.expand_dims(mask[:, :, -1], axis=-1)
        # Since 255 means the 255th class
        # Which doesn't exist, set it to 0 (background)
        mask = tf.where(mask == 255, tf.cast(0, dtype=tf.uint8), mask)

        return {'image': tf_image[:, :, :3], 'segmentation_mask': mask}

    @tf.function
    def prepare_image(self, img_path: tf.string, train=False) -> tuple:
        """
        Load image and if train is True, apply random transformations to the image for training.
        :param img_path: path to image for training
        :param train: whether or not to apply random transformations.
        :return: input image, mask image as tensors
        """

        datapoint = self.parse_image_mask_pair(img_path)
        input_image = datapoint['image']
        input_mask = datapoint['segmentation_mask']

        if train:
            def lr_flip(): return tf.image.flip_left_right(input_image), tf.image.flip_left_right(input_mask)

            def ud_flip(): return tf.image.flip_up_down(input_image), tf.image.flip_up_down(input_mask)

            def rot_left(): return tf.image.rot90(input_image, -1), tf.image.rot90(input_mask, -1)

            def rot_right(): return tf.image.rot90(input_image, 1), tf.image.rot90(input_mask, 1)

            def nothing(): return input_image, input_mask

            input_image, input_mask = tf.case(
                [(tf.logical_and(tf.greater_equal(self.image_modification_random, 0.0), tf.less(self.image_modification_random, 0.2)), lr_flip),
                 (tf.logical_and(tf.greater_equal(self.image_modification_random, 0.2), tf.less(self.image_modification_random, 0.4)), ud_flip),
                 (tf.logical_and(tf.greater_equal(self.image_modification_random, 0.4), tf.less(self.image_modification_random, 0.6)), rot_left),
                 (tf.logical_and(tf.greater_equal(self.image_modification_random, 0.6), tf.less(self.image_modification_random, 0.8)), rot_right),
                 (tf.logical_and(tf.greater_equal(self.image_modification_random, 0.8), tf.less_equal(self.image_modification_random, 1.0)), nothing)])

        input_image = tf.image.resize_with_crop_or_pad(input_image, self.crop_pad_size[1], self.crop_pad_size[0])
        input_mask = tf.cast(tf.image.resize_with_crop_or_pad(input_mask, self.crop_pad_size[1], self.crop_pad_size[0]),
                             dtype=tf.uint8)
        input_image = tf.image.resize(input_image, tf.transpose(self.image_size))
        input_mask = tf.cast(tf.image.resize(input_mask, tf.transpose(self.image_size)), dtype=tf.uint8)

        if self.one_hot:
            one_hot_encoding = tf.expand_dims(tf.reduce_all(tf.equal(input_mask, tf.cast(1, dtype=tf.uint8)), axis=-1), axis=-1)
            for i in range(2, self.num_classes+1):
                tf.autograph.experimental.set_loop_options(shape_invariants=[(one_hot_encoding, tf.TensorShape([one_hot_encoding.shape[0], one_hot_encoding.shape[1], None]))])
                class_map = tf.expand_dims(tf.reduce_all(tf.equal(input_mask, tf.cast(i, dtype=tf.uint8)), axis=-1), axis=-1)
                one_hot_encoding = tf.concat([one_hot_encoding, class_map], axis=-1)
        else:
            one_hot_encoding = input_mask

        # normalize input image
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, one_hot_encoding

    @tf.function
    def count_data_elems(self):
        """
        :return: number of elements in the dataset
        """
        i = self._count_constant
        for _ in self.dataset_filenames:
            i += 1
        return i

    def prepare_dataset(self, dataset, batch_size, shuffle=False, shuffle_seed=None):
        """

        :param dataset:
        :param batch_size:
        :param shuffle:
        :param shuffle_seed:
        :return:
        """
        if shuffle:
            if shuffle_seed is not None:
                dataset = dataset.shuffle(self.count_data_elems(), seed=shuffle_seed)

        dataset = dataset.batch(batch_size)

        # Use buffered prefeching on all datasets
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def get_prepared_datasets(self, shuffle=False, augment=False, fraction_train=0.7, batch_size=32, shuffle_seed=None):
        """

        :param shuffle_seed:
        :param shuffle:
        :param augment:
        :param fraction_train:
        :param batch_size:
        :return:
        """
        tf.random.set_seed(shuffle_seed)
        self.image_modification_random = tf.random.uniform([], seed=shuffle_seed)

        if shuffle:
            self.dataset_filenames = self.dataset_filenames.shuffle(self.count_data_elems(), seed=shuffle_seed)
        num_elems = tf.cast(self.count_data_elems(), dtype=tf.float32)
        train_size = tf.cast(fraction_train * num_elems, dtype=tf.int64)
        val_size = tf.cast((1-fraction_train) * num_elems, dtype=tf.int64)

        train_dataset = self.dataset_filenames.take(train_size)
        train_dataset = train_dataset.map(lambda x: self.prepare_image(x, train=augment))
        train_dataset = self.prepare_dataset(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataset = self.dataset_filenames.skip(train_size)
        val_dataset = val_dataset.take(val_size)
        val_dataset = val_dataset.map(lambda x: self.prepare_image(x, train=False))
        val_dataset = self.prepare_dataset(val_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, val_dataset


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    data = EndoDataset(num_classes=11, image_size=(512, 512), one_hot_encode=False)
    train_ds, validate_ds = data.get_prepared_datasets(shuffle=True, shuffle_seed=1000, batch_size=1)
    for image, label in validate_ds:
        print(image.shape, label.shape)

    for _ in range(2):
        image, label = next(iter(train_ds))
        img = image[0].numpy()
        lbl = label[0].numpy()
        if data.one_hot:
            lbl = np.argmax(lbl == 1, axis=-1)
        print(lbl.max())
        plt.figure()
        _ = plt.imshow(img)
        plt.figure()
        _ = plt.imshow(lbl)

    plt.show(block=True)