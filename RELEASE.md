# Release Guide for TranCIT: Transient Causal Interaction Toolbox

This document provides instructions for releasing new versions of the TranCIT package.

## Release Process

### 1. Pre-Release Checklist

- [ ] All tests pass: `pytest`
- [ ] Code is properly formatted: `black trancit/ tests/`
- [ ] Linting passes: `flake8 trancit/ tests/`
- [ ] Type checking passes: `mypy trancit/`
- [ ] Documentation builds successfully: `cd docs && make html`
- [ ] CHANGELOG.md is updated with new version
- [ ] All dependencies are up to date
- [ ] Security vulnerabilities addressed

### 2. Version Management

This package uses [setuptools_scm](https://github.com/pypa/setuptools_scm) for automatic version management based on Git tags.

#### Version Format

- **Development**: `0.1.0-dev` (no tags)
- **Release**: `0.1.0` (tagged)
- **Post-release**: `0.1.0.post1` (commits after tag)

#### Creating a Release

1. **Update CHANGELOG.md**:

   ```markdown
      ## [0.2.0] - 2025-XX-XX
      
      ### Added
      - New feature descriptions
      
      ### Changed  
      - Changes to existing functionality
      
      ### Fixed
      - Bug fixes
   ```

2. **Create and push a Git tag**:

   ```bash
      git tag -a v0.2.0 -m "Release version 0.2.0"
      git push origin v0.2.0
   ```

3. **Verify version generation**:

   ```bash
      python -c "import trancit; print(trancit.__version__)"
   ```

### 3. Build and Test Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

# Test the built package
twine check dist/*

# Test installation (optional)
pip install dist/trancit-*.whl
```

### 4. Upload to PyPI

#### Test PyPI (recommended first)

```bash
   twine upload --repository testpypi dist/*
```

#### Production PyPI

```bash
   twine upload dist/*
```

### 5. Post-Release

- [ ] GitHub Release created with release notes
- [ ] Documentation deployed to ReadTheDocs
- [ ] Zenodo DOI updated (if applicable)
- [ ] Social media announcements (if applicable)
- [ ] Update package badges in README.md

## Version Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards compatible manner
- **PATCH** version when you make backwards compatible bug fixes

### Examples

- `1.0.0`: First stable release
- `1.1.0`: New features added (backwards compatible)
- `1.1.1`: Bug fixes only
- `2.0.0`: Breaking changes (incompatible API changes)

## Hotfix Releases

For critical bug fixes:

1. Create a hotfix branch from the release tag:

   ```bash
   git checkout -b hotfix/v1.0.1 v1.0.0
   ```

2. Make the necessary fixes

3. Tag and release:

   ```bash
      git tag -a v1.0.1 -m "Hotfix release 1.0.1"
      git push origin v1.0.1
   ```

## CI/CD Integration

GitHub Actions automatically:

- Runs tests on all supported Python versions
- Builds and publishes to PyPI when tags are pushed
- Builds documentation and deploys to ReadTheDocs

## Troubleshooting

### Version Not Updating

- Ensure you've pushed the Git tag: `git push origin --tags`
- Check that setuptools_scm is installed: `pip install setuptools_scm`
- Verify Git repository state: `git status`

### Build Failures

- Clean build directories: `rm -rf dist/ build/ *.egg-info/`
- Update build dependencies: `pip install --upgrade build twine setuptools`
- Check for syntax errors: `python -m py_compile trancit/*.py`

### PyPI Upload Issues

- Verify credentials: `twine check dist/*`
- Check package name availability on PyPI
- Ensure version hasn't been uploaded before

## Support

For release-related issues:

- **Documentation**: `https://trancit.readthedocs.io`
- **Issues**: `https://github.com/CMC-lab/TranCIT/issues`
- **Maintainer**: Salar Nouri (`salr.nouri@gmail.com`)
