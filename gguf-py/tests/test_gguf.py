import unittest
import os
import tempfile
from gguf import GGUFWriter
import numpy as np
import struct

class TestGGUFWriter(unittest.TestCase):
    def test_add_uint8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test0.gguf")
            gguf_writer = GGUFWriter(file_path, "gpt2")
            gguf_writer.add_uint8("test_uint8", 42)
            gguf_writer.write_kv_data_to_file()
            self.assertTrue(os.path.exists(file_path))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test0.gguf")))

            with open(file_path, "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_uint8" in file_content)
                self.assertTrue(b"\x2a" in file_content)

    def test_add_int8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test1.gguf"), "gpt2")
            gguf_writer.add_int8("test_int8", -12)
            gguf_writer.write_kv_data_to_file()
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test1.gguf")))

            with open(os.path.join(temp_dir, "test1.gguf"), "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_int8" in file_content)
                self.assertTrue(b"\xf4" in file_content)

    def test_add_uint16(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test2.gguf"), "gpt2")
            gguf_writer.add_uint16("test_uint16", 65535)
            gguf_writer.write_kv_data_to_file()
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test2.gguf")))

            with open(os.path.join(temp_dir, "test2.gguf"), "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_uint16" in file_content)
                self.assertTrue(b"\xff\xff" in file_content)

    def test_write_tensors_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test3.gguf"), "gpt2")
            tensor_data = np.full((10, 10), 1.0, dtype=np.float32)  # Create a tensor filled with 1.0
            gguf_writer.add_tensor("test_tensor", tensor_data)
            gguf_writer.write_tensors_to_file()
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test3.gguf")))

            with open(os.path.join(temp_dir, "test3.gguf"), "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_tensor" in file_content)
                float_bytes = struct.pack("<f", 1.0)  # Pack float 1.0 into bytes in little-endian format
                self.assertTrue(float_bytes in file_content)

    def test_add_description(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test4.gguf"), "gpt2")
            gguf_writer.add_description("A test model description")
            gguf_writer.write_kv_data_to_file()
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test4.gguf")))

            with open(os.path.join(temp_dir, "test4.gguf"), "rb") as f:
                file_content = f.read()
                self.assertTrue(b"A test model description" in file_content)

    def test_add_uint32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test5.gguf"), "gpt2")
            gguf_writer.add_uint16("test_uint16", 65535)
            gguf_writer.write_kv_data_to_file()
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test5.gguf")))

            with open(os.path.join(temp_dir, "test5.gguf"), "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_uint16" in file_content)
                self.assertTrue(b"\xff\xff" in file_content)

    def test_add_float32(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf_writer = GGUFWriter(os.path.join(temp_dir, "test6.gguf"), "gpt2")
            value_to_write = 3.1415926
            gguf_writer.add_float32("test_float32", value_to_write)
            gguf_writer.write_kv_data_to_file()
            file_path = os.path.join(temp_dir, "test6.gguf")
            self.assertTrue(os.path.exists(file_path))

            with open(file_path, "rb") as f:
                file_content = f.read()
                self.assertTrue(b"test_float32" in file_content)
                expected_float_binary = struct.pack('f', value_to_write)
                self.assertTrue(expected_float_binary in file_content)


if __name__ == '__main__':
    unittest.main()

