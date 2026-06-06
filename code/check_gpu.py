import os
import sys
import re

def check_tensorflow_gpu():
	print("="*60)
	print("TENSORFLOW GPU CHECK (TF 1.15)")
	print("="*60)
	try:
		import tensorflow as tf
		print("TensorFlow version:", tf.__version__)
		print("Built with CUDA:", tf.test.is_built_with_cuda())
		try:
			gpu_available = tf.test.is_gpu_available()
			print("GPU available:", gpu_available)
		except Exception as e:
			print("GPU check error:", e)
			gpu_available = False
		try:
			from tensorflow.python.client import device_lib
			devices = device_lib.list_local_devices()
			print()
			print("All devices:")
			gpu_count = 0
			cpu_count = 0
			for d in devices:
				print(" ", d.name, "-", d.device_type)
				if d.device_type == 'GPU':
					gpu_count += 1
				elif d.device_type == 'CPU':
					cpu_count += 1
			print()
			print("Summary:", gpu_count, "GPU(s),", cpu_count, "CPU(s)")
			return gpu_count > 0
		except ImportError:
			print("Cannot list devices")
			return False
	except ImportError as e:
		print("TensorFlow not installed:", e)
		return False
	except Exception as e:
		print("Error:", e)
		return False

def test_gpu_compute():
	print()
	print("="*60)
	print("GPU COMPUTE TEST")
	print("="*60)
	try:
		import tensorflow as tf
		if tf.test.is_gpu_available():
			with tf.device('/gpu:0'):
				a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
				b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
				c = tf.matmul(a, b)
				print("GPU compute successful!")
				print("Result:", c.numpy())
			return True
		else:
			print("No GPU - skipping test")
			return False
	except Exception as e:
		print("GPU test failed:", e)
		return False

def check_embedding_size():
	print()
	print("="*60)
	print("EMBEDDING SIZE CHECK")
	print("="*60)
	parser_path = os.path.join(os.path.dirname(__file__), 'utility', 'parser.py')
	if os.path.exists(parser_path):
		with open(parser_path, 'r') as f:
			content = f.read()
		match = re.search(r"embed_size.*?default\s*=\s*(\d+)", content)
		if match:
			embed_size = int(match.group(1))
			print("Default embed_size:", embed_size)
		else:
			embed_size = 64
			print("Default embed_size: 64")
		return embed_size
	return 64

def check_environment():
	print()
	print("="*60)
	print("ENVIRONMENT CHECK")
	print("="*60)
	print("Python version:", sys.version.split()[0])
	print("Python executable:", sys.executable)
	print("Platform:", sys.platform)
	conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda')
	print("Conda env:", conda_env)
	print()
	print("Key packages:")
	packages = ['tensorflow', 'numpy', 'scipy', 'matplotlib', 'pandas', 'cython']
	for pkg in packages:
		try:
			mod = __import__(pkg)
			ver = getattr(mod, '__version__', 'unknown')
			print(" ", pkg, ":", ver)
		except ImportError:
			print(" ", pkg, ": NOT INSTALLED")

def main():
	print("="*60)
	print("RecEraser Environment Check")
	print("="*60)
	has_gpu = check_tensorflow_gpu()
	if has_gpu:
		test_gpu_compute()
	embed_size = check_embedding_size()
	check_environment()
	print()
	print("="*60)
	print("SUMMARY")
	print("="*60)
	print("GPU available:", "YES" if has_gpu else "NO (CPU only)")
	print("Embedding size:", embed_size)
	if has_gpu:
		print()
		print("[GPU] Recommended command:")
		print(' python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size', embed_size, '--lr 0.05 --regs "[0.01]" --batch_size 256')
	else:
		print()
		print("[CPU] Recommended command:")
		print(' python RecEraser_BPR.py --dataset ml-1m --part_type 2 --part_num 5 --epoch 30 --embed_size', embed_size, '--lr 0.05 --regs "[0.01]" --batch_size 256')

if __name__ == '__main__':
	main()
