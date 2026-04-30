# CS 4350/7350 — Phase 2: SMS Spam Detection with Apache Spark

A binary text-classification project. The full machine-learning workflow 
is implemented as a series of Spark ML Pipelines, with TF-IDF feature 
engineering and six different classifiers compared side by side.

**Dataset:** [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## Repository layout

```
CS4350_Phase2/
├── notebooks/
│   └── CS4350_7350_Phase2_Spark_Foundation.ipynb   # main notebook
├── data/
│   └── spam.csv                                    # dataset (download from Kaggle)
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook auto-discovers `spam.csv` whether you launch it from the repo
root or from inside `notebooks/` — no path edits required.

---

## Prerequisites

You need **two** specific runtime versions installed. Other versions will
fail with confusing errors, so don't skip this section.

### Python 3.11.9

PySpark 3.5.1 doesn't run on Python 3.12 or newer (the `distutils` module
was removed). Install Python 3.11.9, the last 3.11 release with a binary
installer for Windows and macOS.

- **Download:** https://www.python.org/downloads/release/python-3119/
- **Windows:** during install, check **"Add python.exe to PATH"**.
- **macOS:** install creates the `python3.11` command on your PATH.
- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv`

Verify:

```bash
# Windows (Git Bash or PowerShell)
py -3.11 --version

# macOS / Linux
python3.11 --version
```

Both should print `Python 3.11.9`.

### Java 17 (Eclipse Temurin / OpenJDK)

PySpark 3.5.1 requires Java 8, 11, or 17. **Java 21 and newer will fail**
with `UnsupportedOperationException: getSubject is not supported`. Use
Java 17 (LTS).

- **Download:** https://adoptium.net/temurin/releases/?version=17
- **Windows:** pick the `.msi` installer. On the Custom Setup screen,
  enable **"Set or override JAVA_HOME variable"** and **"Modify PATH
  variable"** (the others are optional).
- **macOS:** pick the `.pkg` installer. It sets `JAVA_HOME` automatically.
- **Linux (Ubuntu/Debian):** `sudo apt install openjdk-17-jdk`

Verify in a **new** terminal (so it picks up the updated PATH):

```bash
java -version       # should start with: openjdk version "17.0.x"
echo $JAVA_HOME     # should contain a path with jdk-17 in it
```

---

## First-time setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd CS4350_Phase2
```
### 2. Create a virtual environment with Python 3.11

```bash
# Windows (Git Bash)
py -3.11 -m venv .venv
source .venv/Scripts/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

After activation, confirm:

```bash
python --version    # must say 3.11.9
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PySpark, numpy, pandas, matplotlib, and Jupyter.

---

## Running the notebook

### In VS Code

1. Open the `CS4350_Phase2` folder directly (**File → Open Folder**), not
   a parent or `notebooks/` subfolder. VS Code only auto-discovers `.venv`
   when it sits at the workspace root.
2. Install the **Python** and **Jupyter** extensions if you don't have them.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry
   labeled `Python 3.11.9 ('.venv': venv)`.
4. Open `notebooks/CS4350_7350_Phase2_Spark_Foundation.ipynb`.
5. In the top-right corner, click the kernel selector → **Select Another
   Kernel** → **Python Environments** → pick the `.venv` one.
6. Run All.

### In Jupyter

```bash
# from the repo root, with .venv active
jupyter notebook notebooks/CS4350_7350_Phase2_Spark_Foundation.ipynb
```

### Verifying the kernel is correct

If you hit a "module not found" error for something that's clearly in
`requirements.txt`, the notebook is almost certainly running on the wrong
Python. Add a temporary cell at the top and run:

```python
import sys
print(sys.executable)
print(sys.version)
```

The path **must** contain `.venv` and the version must start with `3.11.9`.
If not, switch the kernel as described in step 5 above.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'distutils'`

Your kernel is on Python 3.12 or newer. Switch to the `.venv` kernel
running 3.11.9.

### `Py4JJavaError: ... getSubject is not supported`

Your Java is 21 or newer. Install Java 17 from Adoptium and confirm with
`java -version` in a fresh terminal. Restart VS Code completely after
installing so it picks up the new `JAVA_HOME`.

### `Py4JJavaError: ... winutils.exe` (Windows only)

Spark on Windows occasionally needs `winutils.exe` from a Hadoop
distribution. Download from
https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe
into `C:\hadoop\bin\`, then set environment variables:

```powershell
[Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\hadoop\bin", "User")
```

Restart VS Code after setting these.

### `Could not find spam.csv`

The notebook checks `data/spam.csv`, `../data/spam.csv`, and `spam.csv` in
that order. If none exist, either you haven't downloaded the dataset or
you put it in the wrong place. The file should sit at
`CS4350_Phase2/data/spam.csv`.

### Notebook merge conflicts

Jupyter notebooks are JSON and merge poorly. To minimize pain:

- Coordinate with teammates so only one person edits the notebook at a time.
- Always **Kernel → Restart and Clear All Outputs** before committing.
  This strips embedded image data and makes diffs much smaller.
- `git pull` before you start working, `git push` as soon as you finish.

---

## Project structure

The notebook is organized into ten sections that follow the rubric's
expected workflow:

| Section | Contributor | What it does |
|---|---|---|
| 1. Load dataset | Viet | Reads `spam.csv` into a Spark DataFrame |
| 2. EDA | Viet | Schema, class distribution, nulls, duplicates |
| 3. Cleaning | Viet | Drops garbage columns, nulls, duplicates; message-length and top-words analysis |
| 4. Feature engineering | Viet | Builds `text_pipeline_stages` (StringIndexer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF) |
| 4.1 Train/test split | Viet | Stratified 80/20 |
| 4.2 Smoke test | Viet | One LR pipeline to verify wiring |
| 5. Six classifier pipelines | TBD | LR, NB, DT, RF, GBT, SVM |
| 6. Evaluation | TBD | Accuracy, precision, recall, F1, AUC |
| 7. Confusion matrices | TBD | For top 2 models |
| 8. Hyperparameter tuning | TBD | CrossValidator + ParamGridBuilder |
| 9. Model comparison | TBD | Results table + bar chart |
| 10. Discussion | TBD | Interpretation, limitations, conclusions |

---

## License

The dataset is the
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
from the UCI Machine Learning Repository, originally published by Almeida,
Hidalgo & Yamakami (2011).
