# setup_package.py
import os

# Create necessary directories
dirs = ["graphsum", "graphsum/summarizers", "graphsum/evaluation", "graphsum/io"]
for directory in dirs:
    os.makedirs(directory, exist_ok=True)

# Create __init__.py files
for directory in dirs:
    init_file = os.path.join(directory, "__init__.py")
    with open(init_file, "w") as f:
        f.write('"""' + directory.replace("/", ".") + ' package."""\n')
    print(f"Created {init_file}")

print("Package structure created!")