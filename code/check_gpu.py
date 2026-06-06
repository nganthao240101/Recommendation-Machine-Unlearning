"""
Check GPU and embedding size for RecEraser (TF 1.15 compatible)
"""
import os
import sys
import re

def check_tensorflow_gpu():
 """Check if TensorFlow can use GPU (TF 1.15)"""
 print("="*60)
 print("TENSORFLOW GPU CHECK (TF 1.15)")
 print("="*60)

 try:
 import tensorflow as tf
 print(f"TensorFlow version: {tf.__version__}")
 print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

 # TF 1.15 API
 try:
 gpu_available = tf.test.is_gpu_available()
 print(f"GPU available: {gpu_available}")
 except Exception as e:
 print(f"GPU check error: {e}")
 gpu_available = False

 # List devices (TF 1.15 way)
 try:
 from tensorflow.python.client import device_lib
 devices = device_lib.list_local_devices()
 print(f"\nAll devices:")
 gpu_count = 0
 cpu_count = 0
 for d in devices:
 print(f" {d.name} - {d.device_type}")
 if d.device_type == 'GPU':
 gpu_count += 1
 elif d.device_type == 'CPU':
 cpu_count += 1
 print(f"\nSummary: {gpu_count} GPU(s), {cpu_count} CPU(s)")
 return gpu_count > 0

 except ImportError:
 print("Cannot list devices")
 return False

 except ImportError as e:
 print(f"TensorFlow not installed: {e}")
 return False
 except Exception as e:
 print(f"Error: {e}")
 return False

def test_gpu_compute():
 """Test if GPU can do computation"""
 print("\n" + "="*60)
 print("GPU COMPUTE TEST")
 print("="*60)

 try:
 import tensorflow as tf

 if tf.test.is_gpu_available():
 with tf.device('/gpu:0'):
 a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
 b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
 c = tf.matmul(a, b)
 print(f"GPU compute successful!")
 print(f"Result: {c.numpy()}")
 return True
 else:
 print("No GPU - skipping test")
 return False

 except Exception as e:
 print(f"GPU test failed: {e}")
 return False

def check_embedding_size():
 """Check embedding size from parser.py"""
 print("\n" + "="*60)
 print("EMBEDDING SIZE CHECK")
 print("="*60)

 parser_path = os.path.join(os.path.dirname(__file__), 'utility', 'parser.py')
 if os.path.exists(parser_path):
 with open(parser_path, 'r') as f:
 content = f.read()

 match = re.search(r"embed_size.*?default\s*=\s*(\d+)", content)
 if match:
 embed_size = int(match.group(1))
 print(f"Default embed_size: {embed_size}")

 # Check in model
 receraser_path = os.path.join(os.path.dirname(__file__), 'RecEraser_BPR.py')
 if os.path.exists(receraser_path):
 with open(receraser_path, 'r') as f:
 content = f.read()

 if 'args.embed_size' in content:
 print(f"Model uses: args.embed_size = {embed_size}")
 print(f"Embedding matrix shape: [{self.n_users}, {self.num_local}, {embed_size}]")
 print(f"Embedding matrix shape: [{self.n_items}, {self.num_local}, {embed_size}]")
 return embed_size
 else:
 return 64
 else:
 return 64

def check_environment():
 """Check Python and conda environment"""
 print("\n" + "="*60)
 print("ENVIRONMENT CHECK")
 print("="*60)
 print(f"Python version: {sys.version.split()[0]}")
 print(f"Python executable: {sys.executable}")
 print(f"Platform: {sys.platform}")

 conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda')
 print(f"Conda env: {conda_env}")

 # Check CUDA
 cuda_path = os.environ.get('CUDA_HOME', 'Not set')
 print(f"CUDA_HOME: {cuda_path}")

 # Check LD_LIBRARY_PATH
 ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
 print(f"LD_LIBRARY_PATH: {ld_path}")

 print("\nKey packages:")
 packages = ['tensorflow', 'numpy', 'scipy', 'matplotlib', 'pandas', 'cython']
 for pkg in packages:
 try:
 mod = __import__(pkg)
 ver = getattr(mod, '__version__', 'unknown')
 print(f" {pkg}: {ver}")
 except ImportError:
 print(f" {pkg}: NOT INSTALLED")

 # Check protobuf (TF 1.15 needs <=3.20)
 try:
 import google.protobuf
 print(f" protobuf: {google.protobuf.__version__}")
 if int(google.protobuf.__version__.split('.')[0]) > 3:
 print(" WARNING: protobuf > 3.20 may cause issues with TF 1.15")
 except:
 pass

def main():
 print("\n" + "="*60)
 print("RecEraser Environment Check")
 print("="*60)

 has_gpu = check_tensorflow_gpu()
 if has_gpu:
 test_gpu_compute()
 embed_size = check_embedding_size()
 check_environment()

 print("\n" + "="*60)
 print("SUMMARY")
 print("="*60)
 print(f"GPU available: {'YES' if has_gpu else 'NO (CPU only)'}")
 print(f"Embedding size: {embed_size}")

 if has_gpu:
 print(f"\n[GPU] Recommended command:")
 print(f" python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size {embed_size} --lr 0.05 --regs \"[0.01]\" --batch_size 256")
 else:
 print(f"\n[CPU] Recommended command (slower):")
 print(f" python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size {embed_size} --lr 0.05 --regs \"[0.01]\" --batch_size 256")
 print(f"\nExpected: ~20-30 min on CPU, ~5-10 min on GPU")

if __name__ == '__main__':
 main()