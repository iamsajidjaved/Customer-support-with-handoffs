# Customer Support Chatbot with State Machine Pattern

An AI-powered customer support chatbot that demonstrates the state machine pattern using LangChain and Google's Gemini AI. The chatbot intelligently guides users through a structured support workflow for device issues.

## 🌟 Features

- **State Machine Architecture**: Conversation flows through predefined stages (Warranty → Issue Classification → Resolution)
- **Dynamic Prompt Engineering**: Agent behavior automatically adapts based on conversation stage
- **Tool Calling**: AI autonomously invokes functions to record information and transition between states
- **Conversation Memory**: Full conversation history is preserved across interactions
- **User-Friendly Interface**: Clean terminal-based chat interface with helpful error messages

## 🎯 Workflow Stages

### 1. **Warranty Collection**
- Greets the customer
- Determines if device is under warranty
- Records warranty status

### 2. **Issue Classification**
- Asks customer to describe their problem
- Classifies as hardware (physical damage) or software (apps, performance)
- Records issue type

### 3. **Resolution**
- **Software Issues**: Provides troubleshooting steps
- **Hardware Issues**: 
  - In Warranty → Explains warranty repair process
  - Out of Warranty → Escalates to human support for paid repairs

## 📋 Prerequisites

- Python 3.10 or higher
- Google API Key (Gemini API)
- Internet connection

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd "Customer support with handoffs"
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 🔑 Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

## ⚙️ Configuration

Set your Google API key as an environment variable:

**PowerShell (Temporary):**
```powershell
$env:GOOGLE_API_KEY = "your-api-key-here"
```

**Or edit the code directly** (line 50 in `app_interactive.py`):
```python
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
```

## 🎮 Usage

Run the interactive chatbot:

```powershell
python app_interactive.py
```

### Example Conversation

```
You: Hi, my phone screen is broken

Agent: Hi there! I'm sorry to hear about your phone screen. 
       Is your device still under warranty?

You: Yes, it's still under warranty

Agent: Thank you for confirming! Now, can you tell me more about 
       the issue? Is it a physical problem with the screen (like 
       cracks or damage) or a software issue?

You: The screen is physically cracked

Agent: I understand you have a cracked screen and your device is 
       under warranty. Since this is physical damage under warranty, 
       you should be able to get it repaired or replaced...
```

Type `quit`, `exit`, or `q` to end the conversation.

## 📁 Project Structure

```
Customer support with handoffs/
├── app_interactive.py      # Main interactive chatbot application
├── app.py                  # Original demo version with hardcoded examples
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── venv/                  # Virtual environment (created during setup)
```

## 🛠️ Technical Architecture

### State Management
- Uses `AgentState` to track conversation context
- Three possible states: `warranty_collector`, `issue_classifier`, `resolution_specialist`
- State transitions are triggered by tool calls

### Tools (Functions AI Can Call)
- `record_warranty_status()` - Records warranty and moves to issue classification
- `record_issue_type()` - Records issue type and moves to resolution
- `provide_solution()` - Delivers solutions to customer
- `escalate_to_human()` - Escalates complex cases

### Middleware Pattern
- `apply_step_config()` middleware dynamically injects:
  - Step-specific system prompts
  - Available tools for current step
  - State validation

### Memory
- Uses `InMemorySaver` checkpoint to maintain conversation history
- Each session has a unique thread ID

## 🔧 Customization

### Change AI Model

Edit line 54 in `app_interactive.py`:
```python
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# Or try: "gemini-1.5-pro" for more advanced reasoning
```

### Modify Prompts

Edit the prompt constants in Section 5 (lines 197-234):
- `WARRANTY_COLLECTOR_PROMPT`
- `ISSUE_CLASSIFIER_PROMPT`
- `RESOLUTION_SPECIALIST_PROMPT`

### Add New Steps

1. Add new step to `SupportStep` type (line 63)
2. Add state fields to `SupportState` class (lines 66-79)
3. Create new tools if needed (Section 4)
4. Add prompt for new step (Section 5)
5. Update `STEP_CONFIG` dictionary (Section 6)

## 🐛 Troubleshooting

### Rate Limit Errors (429)
- **Issue**: Exceeded free tier quota (15 requests/minute for gemini-1.5-flash)
- **Solution**: Wait 60 seconds or use a different API key

### Permission Denied (403)
- **Issue**: API key is invalid or has been restricted
- **Solution**: Generate a new API key from Google AI Studio

### Import Errors
- **Issue**: Missing dependencies
- **Solution**: Ensure virtual environment is activated and run:
  ```powershell
  pip install -r requirements.txt
  ```

### Model Not Found
- **Issue**: Model name is incorrect
- **Solution**: Use `gemini-1.5-flash` or `gemini-1.5-pro`

## 📚 Learning Resources

### Key Concepts Demonstrated
1. **State Machines**: Structured conversation flows
2. **Middleware Pattern**: Dynamic behavior injection
3. **Tool/Function Calling**: AI-driven function execution
4. **Prompt Engineering**: Context-aware AI instructions
5. **Memory Management**: Stateful conversations

### Code Organization
The code is divided into 9 sections for easy learning:
1. Imports
2. Configuration & Setup
3. State Definition
4. Tool Definitions
5. Prompts for Each Step
6. Step Configuration
7. Middleware (Dynamic Behavior)
8. Agent Creation
9. Interactive Interface

## 📄 Dependencies

- `langchain` - Core LangChain framework
- `langchain-google-genai` - Google Gemini integration
- `langgraph` - State management and graph operations
- `typing-extensions` - Extended type hints

## 📝 License

This project is for educational purposes. Feel free to modify and extend it for your needs.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share improvements

## ✨ Future Enhancements

Potential improvements to consider:
- [ ] Add database persistence instead of in-memory storage
- [ ] Implement webhook notifications for human escalation
- [ ] Add sentiment analysis to detect frustrated customers
- [ ] Support multiple languages
- [ ] Add voice interface
- [ ] Create web UI with Streamlit or Gradio
- [ ] Add analytics dashboard
- [ ] Implement A/B testing for different prompts

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section
2. Review error messages carefully
3. Ensure API key is valid and has quota remaining
4. Verify all dependencies are installed

---

**Author**: Your Name  
**Date**: December 25, 2025  
**Version**: 1.0.0

Made with ❤️ using LangChain and Google Gemini AI
