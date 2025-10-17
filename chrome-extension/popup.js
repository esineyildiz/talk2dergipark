const API_URL = 'http://127.0.0.1:8000';

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
  // Check if keys are saved
  chrome.storage.local.get(['gemini_key', 'openai_key'], (result) => {
    if (result.gemini_key && result.openai_key) {
      document.getElementById('setup').classList.add('hidden');
      document.getElementById('chat').classList.add('active');
    }
  });

  // Auto-resize textarea
  const questionInput = document.getElementById('question');
  if (questionInput) {
    questionInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight) + 'px';
    });

    // Enter to send (Shift+Enter for new line)
    questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('ask-btn').click();
      }
    });
  }

  // Save keys
  document.getElementById('save-keys').addEventListener('click', () => {
    const geminiKey = document.getElementById('gemini-key').value.trim();
    const openaiKey = document.getElementById('openai-key').value.trim();
    
    if (!geminiKey || !openaiKey) {
      alert('Lütfen her iki API anahtarını da girin / Please enter both API keys');
      return;
    }

    chrome.storage.local.set({
      gemini_key: geminiKey,
      openai_key: openaiKey
    }, () => {
      document.getElementById('setup').classList.add('hidden');
      document.getElementById('chat').classList.add('active');
    });
  });

  // Load paper
  document.getElementById('load-paper').addEventListener('click', async () => {
    const loadBtn = document.getElementById('load-paper');
    const paperInfo = document.getElementById('paper-info');
    
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<span class="loading"></span>';
    paperInfo.textContent = 'Makale yükleniyor... / Loading paper...';
    
    // Get current tab URL
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    const paperUrl = tab.url;
    
    chrome.storage.local.get(['gemini_key'], async (result) => {
      try {
        const response = await fetch(`${API_URL}/load_paper`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            paper_url: paperUrl,
            gemini_key: result.gemini_key
          })
        });
        
        const data = await response.json();
        
        if (response.ok) {
          paperInfo.innerHTML = `✅ <strong>Makale yüklendi (${data.chunks} parça) / Paper loaded (${data.chunks} chunks)</strong>`;
          chrome.storage.local.set({current_paper: paperUrl});
          // DON'T hide the button - just change its text
          loadBtn.disabled = false;
          loadBtn.textContent = 'Yeni Makale Yükle / Load New Paper';
        } else {
          paperInfo.textContent = '❌ Hata / Error: ' + (data.detail || data.error || 'Unknown error');
          loadBtn.disabled = false;
          loadBtn.textContent = 'Tekrar Dene / Retry';
        }
      } catch (error) {
        paperInfo.textContent = '❌ Hata / Error: ' + error.message;
        loadBtn.disabled = false;
        loadBtn.textContent = 'Tekrar Dene / Retry';
      }
    });
  });
    

  
  // Ask question
  document.getElementById('ask-btn').addEventListener('click', async () => {
    const question = document.getElementById('question').value.trim();
    if (!question) return;
    
    const messagesDiv = document.getElementById('messages');
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.textContent = question;
    messagesDiv.appendChild(userMsg);
    
    document.getElementById('question').value = '';
    document.getElementById('question').style.height = 'auto';
    
    // Add loading message
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'message assistant loading-msg';
    loadingMsg.innerHTML = '<span class="loading"></span> Düşünüyor... / Thinking...';
    messagesDiv.appendChild(loadingMsg);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    chrome.storage.local.get(['openai_key', 'gemini_key', 'current_paper'], async (result) => {
      try {
        const response = await fetch(`${API_URL}/ask`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            paper_url: result.current_paper,
            question: question,
            openai_key: result.openai_key,
            gemini_key: result.gemini_key
          })
        });
        
        const data = await response.json();
        
        // Remove loading message
        const loadingElement = messagesDiv.querySelector('.loading-msg');
        if (loadingElement) loadingElement.remove();
        
        // Add assistant response
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message assistant';
        assistantMsg.textContent = data.answer || data.error || 'Yanıt alınamadı / No response';
        messagesDiv.appendChild(assistantMsg);
        
      } catch (error) {
        // Remove loading message
        const loadingElement = messagesDiv.querySelector('.loading-msg');
        if (loadingElement) loadingElement.remove();
        
        // Add error message
        const errorMsg = document.createElement('div');
        errorMsg.className = 'message assistant';
        errorMsg.textContent = '❌ Hata / Error: ' + error.message;
        messagesDiv.appendChild(errorMsg);
      }
      
      messagesDiv.scrollTop = messagesDiv.scrollHeight;
    });
  });
});