<div align="center">

# 📧 TempMail Manager

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub issues](https://img.shields.io/github/issues/Sikandar-irfan/Temp-mail.svg)](https://github.com/Sikandar-irfan/Temp-mail/issues)
[![GitHub stars](https://img.shields.io/github/stars/Sikandar-irfan/Temp-mail.svg)](https://github.com/Sikandar-irfan/Temp-mail/stargazers)

*A powerful and secure temporary email solution for your privacy needs*

[Features](#-features) • 
[Installation](#-quick-start) • 
[Usage](#-usage) • 
[Examples](#-code-examples) • 
[Contributing](#-contributing) • 
[License](#-license)

</div>

## ✨ Features

🔒 **Privacy First**
- Generate disposable email addresses instantly
- Protect your real email from spam and tracking
- Auto-delete emails after use

📱 **Smart Monitoring**
- Real-time email notifications
- Live inbox monitoring
- Quick message preview

🌐 **Multiple Providers**
- Support for Guerrilla Mail
- Integration with 1secmail
- Expandable provider system

🛠️ **Power Tools**
- Email forwarding capabilities
- Message backup and export
- Custom email address generation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Sikandar-irfan/Temp-mail.git

# Navigate to project directory
cd Temp-mail

# Run setup script (this will create virtual environment and install dependencies)
chmod +x setup.sh
./setup.sh
```

## 💻 Usage

Start the application:
```bash
python cli.py
```

### Available Commands

| Command | Description |
|---------|-------------|
| 1️⃣ Generate | Create a new temporary email |
| 2️⃣ Monitor | Watch for incoming messages |
| 3️⃣ List | View all active emails |
| 4️⃣ Check | Read received messages |
| 5️⃣ Forward | Forward emails to another address |
| 6️⃣ Export | Save email data |
| 7️⃣ Delete | Remove email addresses |
| 8️⃣ Clear | Clear the screen |
| 9️⃣ Exit | Close the application |

## 📚 Code Examples

### Generate a New Email
```python
from temp_mail_manager import TempMailManager

manager = TempMailManager()
email = manager.generate_email()
print(f"Your temporary email: {email}")
```

### Monitor for New Messages
```python
def on_message(message):
    print(f"New message from: {message['from']}")
    print(f"Subject: {message['subject']}")

manager.monitor_email(email, callback=on_message)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 📜 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

**Key Points:**
- ✅ Free for personal use
- ✅ Must give attribution
- ❌ No commercial use allowed
- ✅ Modifications allowed

## 🌟 Support

Like this project? Please give it a star ⭐ to show your support!

## 📞 Contact

Sikandar Irfan
- GitHub: [@Sikandar-irfan](https://github.com/Sikandar-irfan)

## 🔧 Development

### Running Tests
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=.
```

### Code Style
We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting

```bash
# Format code
black .
isort .

# Check style
flake8
```

---

<div align="center">
Made with ❤️ by Sikandar Irfan
</div>
