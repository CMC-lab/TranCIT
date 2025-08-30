# Security Policy

## Supported Versions

We provide security updates for the following versions of TranCIT:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The TranCIT team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing:

📧 **salr.nouri@gmail.com**

Please include the following information in your report:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** and attack scenarios
4. **Suggested fix** (if you have one)
5. **Your contact information** for follow-up

### Response Timeline

We will acknowledge receipt of your vulnerability report within **48 hours** and will strive to:

- Provide an initial assessment within **5 business days**
- Keep you updated on our progress
- Notify you when the vulnerability is fixed
- Credit you appropriately (unless you prefer to remain anonymous)

### Responsible Disclosure

We practice responsible disclosure and ask that you:

- Give us reasonable time to investigate and fix the issue before making any information public
- Make a good faith effort to avoid privacy violations and disruption of services
- Do not access or modify data that does not belong to you
- Contact us immediately if you believe you have discovered a critical vulnerability

## Security Best Practices

When using TranCIT in your research or production environments:

### Data Handling
- **Sensitive Data**: Be cautious when analyzing sensitive time series data
- **Input Validation**: Always validate input data before processing
- **Temporary Files**: Ensure temporary files are properly cleaned up
- **Memory Management**: Be aware of memory usage with large datasets

### Dependencies
- **Keep Updated**: Regularly update TranCIT and its dependencies to the latest versions
- **Verify Integrity**: Use package checksums when available
- **Audit Dependencies**: Regularly audit your dependency tree for known vulnerabilities

### Environment Security
- **Virtual Environments**: Use virtual environments to isolate package dependencies
- **Access Control**: Implement appropriate access controls for your analysis environment
- **Logging**: Monitor and audit access to sensitive analyses

## Known Security Considerations

### Data Privacy
- TranCIT processes time series data, which may contain sensitive information
- Ensure appropriate data anonymization before analysis when required
- Be aware of potential information leakage through statistical analysis results

### Computational Resources
- Large datasets may consume significant memory and CPU resources
- Implement appropriate resource limits to prevent denial-of-service scenarios
- Monitor computational loads in shared environments

### Reproducibility vs. Privacy
- Statistical analysis results should be reproducible but may reveal information about input data
- Consider differential privacy techniques for sensitive applications
- Document and limit access to intermediate computation results

## Security Updates

Security updates will be:

1. **Announced** in the [CHANGELOG.md](/CHANGELOG.md)
2. **Tagged** with version numbers following semantic versioning
3. **Published** to PyPI as soon as possible
4. **Documented** with migration guidance when necessary

## Contact Information

- **Security Issues**: `salr.nouri@gmail.com`
- **General Issues**: [GitHub Issues](https://github.com/CMC-lab/TranCIT/issues)
- **Maintainer**: Salar Nouri (`salr.nouri@gmail.com`)

## Acknowledgments

We would like to thank the following researchers and security professionals who have helped improve the security of TranCIT:

- *None yet - be the first!*

---

**Thank you for helping keep TranCIT and our community safe!**
