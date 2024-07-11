import unittest
from src.data_loader import load_image, load_json

class TestDataLoader(unittest.TestCase):
    def test_load_image(self):
        image = load_image('path/to/sample.jpg')
        self.assertEqual(image.shape, (height, width, channels))

    def test_load_json(self):
        data = load_json('path/to/sample.json')
        self.assertIn('key', data)

if __name__ == '__main__':
    unittest.main()
