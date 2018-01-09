import mock
import numpy as np
import unittest

from .image_dataset import ImageDataset

class TestImageDataset(unittest.TestCase):
    def test_empty(self):
        dataset = ImageDataset([])
        self.assertEqual(0, len(dataset))

    @mock.patch('laia.data.image_dataset.Image.open')
    def test_simple(self, mock_Image_open):
        expected_image = np.array([[1,2], [3,4]])
        mock_Image_open.return_value = expected_image
        dataset = ImageDataset(['filename.jpg'])
        self.assertEqual(1, len(dataset))
        np.testing.assert_equal(len(dataset[0]), 2)
        np.testing.assert_almost_equal(expected_image, dataset[0]['img'])


if __name__ == '__main__':
    unittest.main()