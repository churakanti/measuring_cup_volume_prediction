# GitHub Setup Guide - Step by Step

This guide will walk you through setting up your Measuring Cup Volume Prediction project on GitHub.

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Git installed on your computer
- ✅ A GitHub account
- ✅ All project files ready (README, code, docs, etc.)

## 🎯 Quick Start

If you're familiar with Git and GitHub, here's the quick version:

```bash
# Navigate to project directory
cd "C:\Users\chura\Downloads\CJ\Collab_2-20250907T043819Z-1-001\Collab_2"

# Initialize Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Measuring cup volume prediction project"

# Create GitHub repository (via web or CLI)
# Then add remote and push
git remote add origin https://github.com/yourusername/measuring-cup-volume-prediction.git
git branch -M main
git push -u origin main
```

## 📖 Detailed Step-by-Step Guide

### Step 1: Verify Git Installation

Open your terminal (Command Prompt, PowerShell, or Git Bash) and check if Git is installed:

```bash
git --version
```

**Expected output**: `git version 2.x.x`

**If Git is not installed**:
- Download from [git-scm.com](https://git-scm.com/)
- Install with default settings
- Restart your terminal

### Step 2: Configure Git (First Time Only)

Set your name and email for Git commits:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify configuration**:
```bash
git config --global --list
```

### Step 3: Navigate to Project Directory

```bash
# Windows (PowerShell/CMD)
cd "C:\Users\chura\Downloads\CJ\Collab_2-20250907T043819Z-1-001\Collab_2"

# Verify you're in the right directory
dir  # Should show README.md, LICENSE, requirements.txt, etc.
```

### Step 4: Initialize Git Repository

```bash
git init
```

**Expected output**: `Initialized empty Git repository in ...`

This creates a hidden `.git` folder in your project directory.

### Step 5: Verify .gitignore is Working

Before adding files, verify that the `.gitignore` file will exclude large datasets:

```bash
# Check what files Git will track
git status

# You should NOT see:
# - BMC_NewResized/ directory
# - BMC_OR/ directory
# - .venv/ directory
# - *.zip files
```

**If you see large folders listed**:
- Check that `.gitignore` file exists
- Verify `.gitignore` is in the root directory
- Make sure `.gitignore` has the correct syntax (no extra spaces)

### Step 6: Stage Files for Commit

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what's been staged
git status
```

**Expected output**: Should show files like:
- `README.md`
- `LICENSE`
- `requirements.txt`
- `BMC_ES_1plus1.py`
- `docs/` directory files
- And other code files

**Should NOT show**:
- Dataset directories (`BMC_NewResized/`, etc.)
- Model files (`*.keras`, `*.h5`)
- Virtual environment (`.venv/`)

### Step 7: Create Initial Commit

```bash
git commit -m "Initial commit: Measuring cup volume prediction project"
```

**Expected output**: Shows files added and insertions made.

### Step 8: Create GitHub Repository

#### Option A: Using GitHub Website (Recommended for Beginners)

1. Go to [github.com](https://github.com) and log in
2. Click the **"+"** icon in top-right corner
3. Select **"New repository"**
4. Fill in repository details:
   - **Repository name**: `measuring-cup-volume-prediction`
   - **Description**: `Deep learning for automated volume measurement prediction from measuring cup images`
   - **Visibility**: Public (for portfolio) or Private
   - **DO NOT** check "Initialize with README" (you already have one)
   - **DO NOT** add .gitignore or license (you already have them)
5. Click **"Create repository"**

#### Option B: Using GitHub CLI (Advanced)

If you have GitHub CLI installed:

```bash
gh repo create measuring-cup-volume-prediction --public --source=. --remote=origin --push
```

### Step 9: Connect Local Repository to GitHub

After creating the GitHub repository, you'll see instructions. Use these commands:

```bash
# Add remote repository
git remote add origin https://github.com/yourusername/measuring-cup-volume-prediction.git

# Verify remote was added
git remote -v
```

**Replace `yourusername`** with your actual GitHub username.

### Step 10: Push to GitHub

```bash
# Rename branch to 'main' (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Expected output**: Shows upload progress and confirms push to GitHub.

### Step 11: Verify on GitHub

1. Go to your repository on GitHub: `https://github.com/yourusername/measuring-cup-volume-prediction`
2. You should see:
   - ✅ README.md displayed on the homepage
   - ✅ All your code files
   - ✅ docs/ directory
   - ✅ LICENSE file
   - ✅ **No** dataset directories (excluded by .gitignore)

## 📊 What Should Be on GitHub

### ✅ SHOULD BE on GitHub:
- Source code (`.py` files)
- Jupyter notebooks (`.ipynb` files)
- Documentation (`.md` files)
- Configuration files (`requirements.txt`)
- Small images/diagrams (in `docs/images/`)
- License and README
- `.gitignore` file itself

### ❌ Should NOT be on GitHub:
- Dataset directories (`BMC_NewResized/`, `BMC_OR/`, etc.)
- Large archive files (`BMC_OR.zip`)
- Trained models (`*.keras`, `*.h5`)
- Virtual environments (`.venv/`, `venv/`)
- Python cache (`__pycache__/`)
- Personal files (`desktop.ini`)

## 🔧 Next Steps After GitHub Setup

### 1. Upload Dataset to External Hosting

Your dataset should be hosted separately:

**Option A: Google Drive**
1. Compress dataset: `zip -r BMC_Dataset.zip BMC_NewResized/`
2. Upload to Google Drive
3. Get shareable link
4. Update `docs/DATASET.md` with the download link

**Option B: Kaggle Datasets**
1. Create account on [kaggle.com](https://www.kaggle.com/)
2. Go to "Datasets" → "New Dataset"
3. Upload your dataset files
4. Add link to `docs/DATASET.md`

**Option C: Zenodo**
1. Create account on [zenodo.org](https://zenodo.org/)
2. Upload dataset and get DOI
3. Add citation to `docs/DATASET.md`

### 2. Update Dataset Links in Documentation

Edit `docs/DATASET.md` and replace placeholder links:

```bash
# Open in your text editor
notepad docs/DATASET.md  # Windows
# or
nano docs/DATASET.md     # Linux/Mac
```

Add your actual download links in the "Download Instructions" section.

### 3. Add GitHub Repository Link to README

Update the clone command in `README.md`:

```bash
# Replace this line:
git clone https://github.com/yourusername/measuring-cup-volume-prediction.git

# With your actual repository URL:
git clone https://github.com/your-actual-username/measuring-cup-volume-prediction.git
```

### 4. Customize Portfolio Elements

**Add your information**:
- Your name in LICENSE file
- Your email in documentation
- Your GitHub username in README
- Your academic/professional affiliations

**Add optional enhancements**:
- Project screenshots in `docs/images/`
- Result visualizations (graphs, accuracy plots)
- Sample prediction images
- Demo GIFs showing the model in action

### 5. Create GitHub Repository Description

On your GitHub repository page:
1. Click **"About"** (gear icon on right side)
2. Add description: `Deep learning for automated volume measurement prediction from measuring cup images using EfficientNet and ES(1+1) optimization`
3. Add topics/tags: `deep-learning`, `computer-vision`, `keras`, `tensorflow`, `efficientnet`, `transfer-learning`, `volume-prediction`, `image-regression`
4. Add website (if you have a portfolio/demo)
5. Click **"Save changes"**

## 📝 Making Future Changes

### Basic Git Workflow

After making changes to your code:

```bash
# Check what changed
git status

# Stage specific files
git add filename.py

# Or stage all changes
git add .

# Commit with descriptive message
git commit -m "Add feature: improved data augmentation"

# Push to GitHub
git push
```

### Good Commit Message Examples

✅ Good:
- `"Fix bug in volume prediction normalization"`
- `"Add documentation for ES(1+1) algorithm"`
- `"Improve EfficientNet model accuracy to 95%"`
- `"Update requirements.txt with latest dependencies"`

❌ Bad:
- `"Fixed stuff"`
- `"Updates"`
- `"asdfasdf"`
- `"Final version 3 (really final this time)"`

## 🎨 Enhancing Your GitHub Repository

### Add Badges to README

Badges make your repository look professional. Add these at the top of your `README.md`:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Keras](https://img.shields.io/badge/Keras-3.11+-red.svg)](https://keras.io/)
```

### Create Project Showcase

Take screenshots of:
- Model training graphs (accuracy, loss curves)
- Prediction results (before/after images)
- Confusion matrix or error analysis
- Architecture diagrams

Save in `docs/images/` and reference in README.

### Add GitHub Actions (Optional)

Create `.github/workflows/test.yml` for automated testing:

```yaml
name: Python Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - run: pip install -r requirements.txt
    - run: python -m pytest tests/
```

## 🐛 Troubleshooting

### Problem: "Repository too large" error

**Solution**: Dataset wasn't excluded by .gitignore
```bash
# Remove large files from Git tracking
git rm -r --cached BMC_NewResized/
git rm -r --cached BMC_OR/
git commit -m "Remove dataset from Git tracking"
git push
```

### Problem: .gitignore not working

**Solution**: Files were already tracked before .gitignore
```bash
# Remove from Git but keep on disk
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules"
git push
```

### Problem: Push rejected (authentication)

**Solution**: Use Personal Access Token
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Problem: Merge conflicts

**Solution**:
```bash
# Pull latest changes first
git pull origin main

# Resolve conflicts in your editor
# Then commit and push
git add .
git commit -m "Resolve merge conflicts"
git push
```

## 📚 Additional Resources

- **Git Documentation**: [git-scm.com/doc](https://git-scm.com/doc)
- **GitHub Guides**: [guides.github.com](https://guides.github.com/)
- **GitHub Learning Lab**: [lab.github.com](https://lab.github.com/)
- **Git Cheat Sheet**: [education.github.com/git-cheat-sheet-education.pdf](https://education.github.com/git-cheat-sheet-education.pdf)

## ✅ Checklist

Before considering your GitHub setup complete:

- [ ] Repository created on GitHub
- [ ] All code files pushed
- [ ] .gitignore working (no datasets on GitHub)
- [ ] README.md displays correctly
- [ ] License file included
- [ ] Documentation in docs/ folder
- [ ] Requirements.txt has clean dependencies
- [ ] Repository description and topics added
- [ ] Dataset hosted externally with download links
- [ ] Personal information updated (name, email, etc.)
- [ ] Repository is public (if for portfolio)
- [ ] Badges added to README (optional)
- [ ] Visual assets included (optional)

## 🎓 Best Practices

1. **Commit often**: Small, frequent commits are better than large, infrequent ones
2. **Descriptive messages**: Write clear commit messages explaining what changed
3. **Keep .gitignore updated**: Add new exclusions as needed
4. **Document changes**: Update README when adding features
5. **Use branches**: Create branches for major features (`git checkout -b feature-name`)
6. **Review before pushing**: Check `git status` and `git diff` before committing

---

**Congratulations!** 🎉 Your project is now on GitHub and ready to showcase to the world!

For any questions or issues, refer to the main [README.md](README.md) or open an issue on GitHub.
