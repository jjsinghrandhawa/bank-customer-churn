import os


PROJECT_NAME = "bank-customer-churn"


folders = [
    "src",
    "src/components",
    "src/pipeline",
    "src/utils",
    "data",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts",
    "artifacts/models",
    "artifacts/preprocessing",
    "logs",
    "notebooks",
    "tests",
]


files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/pipeline/__init__.py",
    "src/utils/__init__.py",
    "src/logger.py",
    "src/exception.py",
    ".gitignore",
    "requirements.txt",
    "README.md",
]


def create_directories():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")


def create_files():
    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass

            print(f"Created file: {file}")

        else:
            print(f"Already exists: {file}")


if __name__ == "__main__":
    print(f"\nCreating project structure: {PROJECT_NAME}\n")

    create_directories()
    create_files()

    print("\nProject structure created successfully.")