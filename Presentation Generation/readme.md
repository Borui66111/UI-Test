# Presentation Generation Program Setup Guide
Note: open this file in a editor that can correctly format .md file. You can find an online md reader to properly display the content of this file.

## About
This markdown file serves as an authoritative instruction to set up a Presentation Generation program in Python. It includes comprehensive steps for installing necessary packages and software, and instructions on how to execute the program. Additionally, it guides on setting up a `.env` file to store the OpenAI API key. You will learn how to download the program from GitHub without needing a GitHub account and how to complete the necessary tasks using the command line.

## Content

### 1. Install Python

First, you need to install Python on your system. Follow these steps to download and install Python:

1. Go to the official Python website: [Python Downloads](https://www.python.org/downloads/)
2. Download the latest version of Python for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the installation instructions. Ensure to check the option to add Python to your PATH during the installation process.

To verify the installation, open the Command Prompt (cmd) or Terminal and type:
```sh
python --version
```
You should see the version of Python you installed.

### 2. Install pip

`pip` is the package installer for Python. It is typically included with Python installations. To check if `pip` is installed, run:
```sh
pip --version
```
If you see the version of `pip`, it is already installed. If not, you can install `pip` by following the instructions on the [pip installation page](https://pip.pypa.io/en/stable/installation/).

### 3. Download the Program from GitHub

To download the program from GitHub without a GitHub account, follow these steps:

1. Navigate to the GitHub repository page of the Presentation Generation program.
2. Click on the "Code" button and select "Download ZIP".
3. Extract the downloaded ZIP file to your desired location.

Alternatively, you can use the following command to download the repository using `wget` (Linux and macOS) or `curl` (Windows):
```sh
# Using wget (Linux and macOS)
wget https://github.com/Borui66111/UI-Test/archive/refs/heads/main.zip -O presentation_program.zip

# Using curl (Windows)
curl -L https://github.com/Borui66111/My-UI/archive/refs/heads/main.zip -o presentation_program.zip

# Extract the ZIP file
unzip presentation_program.zip
```

### 4. Install Required Packages

Navigate to the directory where you extracted the ZIP file using the Command Prompt or Terminal. Then, run the following command to install the required packages:
```sh
pip install -r requirements.txt
```
This command reads the `requirements.txt` file and installs all the necessary Python packages for the program.

### 5. Set Up the `.env` File

The `.env` file is crucial for securely storing sensitive information, such as your OpenAI API key. Follow these steps to set up the `.env` file:

1. In the root directory of your project, create a new file named `.env`. In this case it can be the parent folder or the same folder of the program.
2. Open the `.env` file in a text editor and add the following line, replacing `your_openai_api_key` with your actual OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key
```
3. Save and close the `.env` file.

### 6. How to Run

To run the Presentation Generation program, use the following steps:

1. Ensure you are in the directory where the program files are located.
2. Execute the program using Python:
```sh
cd ./presentation_program/UI-Test-main/Presentation Generation/console
python test.py
```
Replace `test.py` with the actual name of the Python script that starts the program if it is different.

Follow any additional on-screen instructions provided by the program to generate your presentation.

---

By adhering to these detailed instructions, you should be able to set up and run the Presentation Generation program efficiently. If you encounter any issues, refer to the documentation or support resources provided with the program.
