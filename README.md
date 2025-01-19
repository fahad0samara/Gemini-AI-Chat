# 🤖 Gemini AI Chat Application

A powerful and modern web-based chat application powered by Google's Gemini AI model. This application provides an interactive chat interface with advanced features like file sharing, code execution, chat export, and message reactions.

![Gemini AI Chat](https://img.shields.io/badge/AI-Gemini-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### Core Features
- 💬 Real-time chat with Gemini AI
- 📁 File upload and sharing
- 💻 Code execution with syntax highlighting
- 📥 Chat export to PDF and JSON
- 😄 Message reactions with emojis

### AI Features
- 🖼️ Image analysis and enhancement
  - Text detection in images
  - Object detection
  - Image enhancement
  - QR code generation
- 🌍 Multi-language translation
- 📊 Chat summarization
- 🎭 Sentiment analysis

### AI Personas
- 👨‍🏫 Professor AI: Educational explanations
- 👨‍💻 Code Master: Programming assistance
- 🎨 Creative Spirit: Brainstorming and ideas
- 📈 Data Sage: Analysis and insights

## 🏗️ Project Structure

```
.
├── .env                # Environment variables configuration
├── README.md          # Project documentation (you are here)
├── chat_history.db    # SQLite database for chat storage
├── requirements.txt   # Python package dependencies
├── simple_app.py      # Main application code
├── templates/         # HTML templates for web interface
│   ├── index.html    # Main chat interface
│   └── style.css     # Custom styling
└── venv/             # Python virtual environment
```

## 🚀 Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- A Gemini API key from Google

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/Gemini-AI-Chat.git
cd Gemini-AI-Chat
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

5. Initialize the database:
```bash
python simple_app.py --init-db
```

## 🎮 Usage

1. Start the application:
```bash
python simple_app.py
```

2. Open your browser and go to `http://localhost:3693`

### Chat Features
- 💭 Start chatting with the AI
- 📎 Upload files by clicking the upload button or drag-and-drop
- ⚡ Execute code by typing ```python your_code_here```
- 💾 Export chat by clicking the export button
- 👍 React to messages using the reaction buttons
- 🔍 Analyze images by uploading them
- 🌐 Translate messages using the translate button
- 📝 Get chat summaries using the summarize button

### AI Personas
Switch between different AI personas for specialized help:
- Professor AI: For detailed explanations
- Code Master: For programming help
- Creative Spirit: For brainstorming
- Data Sage: For data analysis

## 🛠️ Advanced Features

### Image Analysis
The application supports various image analysis features:
- Text extraction from images
- Object detection
- Image enhancement
- QR code generation for sharing

### Code Execution
- Supports multiple programming languages
- Syntax highlighting
- Error handling
- Output display
- Code sharing

### Chat Export
Export your conversations in multiple formats:
- PDF with formatting
- JSON for data analysis
- Includes timestamps and reactions

## 🔒 Security

- API keys stored securely in environment variables
- Input sanitization
- File upload restrictions
- Secure database operations
- XSS protection

## 🐛 Troubleshooting

Common issues and solutions:

1. **Connection Error**
   - Check your internet connection
   - Verify your API key in .env file
   - Ensure the server is running

2. **File Upload Issues**
   - Check file size limits
   - Verify supported file types
   - Ensure upload directory permissions

3. **Database Errors**
   - Check database file permissions
   - Verify SQLite installation
   - Try reinitializing the database

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - feel free to use this project for your own purposes.

## 👥 Contact

- Create an issue for bug reports
- Submit a pull request for contributions
- Email: your.email@example.com

## 🙏 Acknowledgments

- Google Gemini AI team
- Flask framework
- All contributors

---
Made with ❤️ using Google Gemini AI
