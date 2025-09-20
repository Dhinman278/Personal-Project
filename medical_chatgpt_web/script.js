// Medical AI Assistant - ChatGPT.com Style Web Application
// Interactive JavaScript for modern medical consultation interface

class MedicalChatApp {
    constructor() {
        this.currentChatId = 'current';
        this.conversations = new Map();
        this.isTyping = false;
        this.messageHistory = [];
        this.patientData = null;
        
        this.initializeApp();
        this.setupEventListeners();
        this.loadConversations();
    }

    initializeApp() {
        console.log('🏥 Medical AI Assistant initializing...');
        
        // Initialize DOM elements
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebarToggle'),
            newChatBtn: document.getElementById('newChatBtn'),
            chatContainer: document.getElementById('chatContainer'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            inputForm: document.getElementById('inputForm'),
            suggestions: document.getElementById('suggestions'),
            emergencyModal: document.getElementById('emergencyModal'),
            emergencyModalClose: document.getElementById('emergencyModalClose'),
            settingsModal: document.getElementById('settingsModal'),
            settingsModalClose: document.getElementById('settingsModalClose'),
            settingsBtn: document.getElementById('settingsBtn'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            emergencyActions: document.getElementById('emergencyActions')
        };

        // Initialize conversation
        this.startNewConversation();
    }

    setupEventListeners() {
        // Mobile sidebar toggle
        this.elements.sidebarToggle?.addEventListener('click', () => {
            this.elements.sidebar.classList.toggle('mobile-open');
        });

        // New chat button
        this.elements.newChatBtn.addEventListener('click', () => {
            this.startNewConversation();
        });

        // Message input handling
        this.elements.messageInput.addEventListener('input', () => {
            this.handleInputChange();
        });

        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Form submission
        this.elements.inputForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Suggestion buttons
        this.elements.suggestions.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const suggestion = e.target.dataset.suggestion;
                this.useSuggestion(suggestion);
            }
        });

        // Modal controls
        this.setupModalListeners();

        // Settings button
        this.elements.settingsBtn?.addEventListener('click', () => {
            this.showSettingsModal();
        });

        // Emergency button (mobile)
        document.getElementById('emergencyBtnMobile')?.addEventListener('click', () => {
            this.showEmergencyModal();
        });

        // Chat history items
        this.setupChatHistoryListeners();

        // Auto-resize textarea
        this.setupTextareaAutoResize();
    }

    setupModalListeners() {
        // Emergency modal
        this.elements.emergencyModalClose?.addEventListener('click', () => {
            this.hideModal(this.elements.emergencyModal);
        });

        // Settings modal
        this.elements.settingsModalClose?.addEventListener('click', () => {
            this.hideModal(this.elements.settingsModal);
        });

        // Click outside to close
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                this.hideModal(e.target);
            }
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const activeModal = document.querySelector('.modal-overlay.active');
                if (activeModal) {
                    this.hideModal(activeModal);
                }
            }
        });
    }

    setupChatHistoryListeners() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('.chat-item')) {
                const chatItem = e.target.closest('.chat-item');
                const chatId = chatItem.dataset.chatId;
                this.loadConversation(chatId);
            }
        });
    }

    setupTextareaAutoResize() {
        const textarea = this.elements.messageInput;
        
        textarea.addEventListener('input', () => {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        });
    }

    handleInputChange() {
        const input = this.elements.messageInput.value.trim();
        this.elements.sendBtn.disabled = input.length === 0;
        
        // Check for emergency keywords
        const emergencyKeywords = [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 
            'difficulty breathing', 'severe bleeding', 'unconscious',
            'overdose', 'poisoning', 'emergency', 'help me', 'dying'
        ];
        
        const hasEmergencyKeywords = emergencyKeywords.some(keyword => 
            input.toLowerCase().includes(keyword.toLowerCase())
        );
        
        if (hasEmergencyKeywords) {
            this.showEmergencyActions();
        } else {
            this.hideEmergencyActions();
        }
    }

    showEmergencyActions() {
        this.elements.emergencyActions.style.display = 'flex';
        this.elements.suggestions.style.display = 'none';
    }

    hideEmergencyActions() {
        this.elements.emergencyActions.style.display = 'none';
        this.elements.suggestions.style.display = 'flex';
    }

    startNewConversation() {
        console.log('🆕 Starting new conversation');
        
        // Clear chat messages
        this.elements.chatMessages.innerHTML = '';
        this.elements.chatMessages.style.display = 'none';
        this.elements.welcomeScreen.style.display = 'flex';
        
        // Reset state
        this.currentChatId = 'current';
        this.messageHistory = [];
        this.isTyping = false;
        
        // Clear input
        this.elements.messageInput.value = '';
        this.elements.sendBtn.disabled = true;
        
        // Hide emergency actions
        this.hideEmergencyActions();
        
        // Update active chat item
        document.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const currentChat = document.querySelector('.chat-item[data-chat-id="current"]');
        if (currentChat) {
            currentChat.classList.add('active');
        }
    }

    async sendMessage() {
        const messageText = this.elements.messageInput.value.trim();
        
        if (!messageText || this.isTyping) return;

        console.log('📤 Sending message:', messageText);

        // Hide welcome screen and show chat
        this.elements.welcomeScreen.style.display = 'none';
        this.elements.chatMessages.style.display = 'block';

        // Add user message
        this.addMessage(messageText, 'user');
        
        // Clear input
        this.elements.messageInput.value = '';
        this.elements.sendBtn.disabled = true;
        this.elements.messageInput.style.height = 'auto';
        
        // Hide suggestions and emergency actions
        this.elements.suggestions.style.display = 'none';
        this.hideEmergencyActions();

        // Add to message history
        this.messageHistory.push({ role: 'user', content: messageText });

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get AI response
            const response = await this.getAIResponse(messageText);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add AI response
            this.addMessage(response, 'assistant');
            
            // Add to message history
            this.messageHistory.push({ role: 'assistant', content: response });
            
        } catch (error) {
            console.error('❌ Error getting AI response:', error);
            this.hideTypingIndicator();
            this.addMessage('I apologize, but I encountered an error processing your request. Please try again or contact emergency services if this is urgent.', 'assistant');
        }
    }

    addMessage(content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Format message content
        messageText.innerHTML = this.formatMessageContent(content);
        
        messageContent.appendChild(messageText);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.elements.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.scrollToBottom();
        
        console.log(`💬 Added ${role} message`);
    }

    formatMessageContent(content) {
        // Convert markdown-like formatting
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        // Convert bullet points
        formatted = formatted.replace(/^• (.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/^✅ (.+)$/gm, '<li style="color: var(--success-color);">✅ $1</li>');
        formatted = formatted.replace(/^🚨 (.+)$/gm, '<li style="color: var(--danger-color);">🚨 $1</li>');
        formatted = formatted.replace(/^⚠️ (.+)$/gm, '<li style="color: var(--warning-color);">⚠️ $1</li>');
        
        // Wrap consecutive list items in ul tags
        formatted = formatted.replace(/(<li>.*<\/li>[\s\S]*?)(?=<li>|$)/g, '<ul>$1</ul>');
        
        return formatted;
    }

    showTypingIndicator() {
        this.isTyping = true;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="typing-avatar">AI</div>
            <div class="typing-content">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        this.elements.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async getAIResponse(userMessage) {
        // Simulate processing delay
        await this.delay(2000 + Math.random() * 1000);
        
        const messageLower = userMessage.toLowerCase();
        
        // Emergency detection
        const emergencyKeywords = [
            'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 
            'difficulty breathing', 'severe bleeding', 'unconscious',
            'severe allergic reaction', 'overdose', 'poisoning'
        ];
        
        if (emergencyKeywords.some(keyword => messageLower.includes(keyword))) {
            return this.getEmergencyResponse(userMessage);
        }
        
        // Medical symptom responses
        if (messageLower.includes('fever') || messageLower.includes('temperature') || messageLower.includes('hot')) {
            return this.getFeverResponse();
        }
        
        if (messageLower.includes('headache') || messageLower.includes('head pain')) {
            return this.getHeadacheResponse();
        }
        
        if (messageLower.includes('stomach') || messageLower.includes('nausea') || messageLower.includes('vomit')) {
            return this.getDigestiveResponse();
        }
        
        if (messageLower.includes('medication') || messageLower.includes('medicine') || messageLower.includes('drug')) {
            return this.getMedicationResponse();
        }
        
        if (messageLower.includes('cough') || messageLower.includes('cold') || messageLower.includes('flu')) {
            return this.getRespiratoryResponse();
        }
        
        // General health consultation
        return this.getGeneralResponse(userMessage);
    }

    getEmergencyResponse(userMessage) {
        return `🚨 **MEDICAL EMERGENCY DETECTED**

Your symptoms may indicate a serious medical emergency requiring immediate attention.

**IMMEDIATE ACTIONS:**
• **Call 911 NOW** - Do not delay
• Do not drive yourself to the hospital
• Stay calm and follow dispatcher instructions
• Have someone stay with you if possible

**While waiting for emergency services:**
• Keep airways clear
• Do not take medications unless instructed by emergency personnel
• Gather medical information and current medications
• Ensure door is unlocked for first responders

**Emergency Contacts:**
• **Emergency Services:** 911
• **Poison Control:** 1-800-222-1222

⚠️ **This is an AI assessment. If you believe this is an emergency, call 911 immediately regardless of this analysis.**

Time is critical in medical emergencies. Professional medical care is essential.`;
    }

    getFeverResponse() {
        return `🤒 **Fever Symptom Analysis**

Based on your symptoms, you may be experiencing a fever. Here's what I recommend:

**Immediate Care Steps:**
✅ **Rest** - Get plenty of sleep and avoid strenuous activities
✅ **Hydration** - Drink clear fluids (water, broth, herbal tea)
✅ **Monitor Temperature** - Check every 4 hours with a thermometer
✅ **Fever Reduction** - Consider acetaminophen or ibuprofen as directed

**When to Seek Medical Care:**
🚨 Temperature above 103°F (39.4°C)
🚨 Fever lasting more than 3 days
🚨 Difficulty breathing or chest pain
🚨 Severe dehydration (dizziness, dry mouth, little/no urination)
🚨 Persistent vomiting
🚨 Severe headache or neck stiffness

**COVID-19 Considerations:**
• Consider getting tested for COVID-19
• Isolate from others until fever-free for 24 hours
• Wear a mask around others if isolation isn't possible

**Additional Questions:**
Do you have any other symptoms like headache, body aches, cough, or sore throat? This information can help me provide more specific guidance.

Remember: This is educational information. Always consult healthcare professionals for proper medical evaluation.`;
    }

    getHeadacheResponse() {
        return `🧠 **Headache Assessment**

Headaches can have various causes. Let me help you understand your symptoms:

**Common Headache Types:**
• **Tension headaches** - Pressure around forehead/temples
• **Migraine** - Throbbing pain, often with nausea/light sensitivity
• **Sinus headaches** - Pressure around eyes, nose, cheeks
• **Dehydration headaches** - General head pain with thirst

**Immediate Relief Measures:**
✅ **Hydration** - Drink 16-20 oz of water slowly
✅ **Rest** - Lie down in a dark, quiet room
✅ **Cold/Heat Therapy** - Ice pack or warm compress on head/neck
✅ **Over-the-counter pain relief** - Acetaminophen or ibuprofen as directed

**Emergency Warning Signs - Call 911:**
🚨 Sudden, severe headache ("worst headache of your life")
🚨 Headache with fever, stiff neck, and rash
🚨 Headache after head injury
🚨 Headache with confusion, vision changes, or weakness
🚨 Headache with difficulty speaking

**When to See a Doctor:**
• Frequent headaches (more than 2-3 per week)
• Headaches that worsen over time
• Changes in headache pattern
• Headaches that don't respond to over-the-counter medication

**Questions to Help Assess:**
• How long have you had this headache?
• On a scale of 1-10, how severe is the pain?
• Is this different from headaches you've had before?
• Any associated symptoms (nausea, vision changes, fever)?

Would you like to provide more details about your headache symptoms?`;
    }

    getDigestiveResponse() {
        return `🤢 **Digestive System Assessment**

I understand you're experiencing digestive discomfort. Here's my analysis and recommendations:

**Common Causes:**
• Viral gastroenteritis (stomach flu)
• Food poisoning or foodborne illness
• Dietary indiscretion or new foods
• Medication side effects
• Stress-related digestive upset

**Immediate Treatment (BRAT Diet + Hydration):**
✅ **Clear fluids** - Water, clear broth, electrolyte solutions
✅ **BRAT foods** - Bananas, Rice, Applesauce, Toast
✅ **Small, frequent sips** rather than large amounts at once
✅ **Avoid** - Dairy, caffeine, alcohol, fatty/spicy foods

**Symptom Management:**
• **Nausea:** Ginger tea, small ice chips, crackers
• **Vomiting:** Wait 1-2 hours, then try small sips of clear fluids
• **Diarrhea:** Stay hydrated, avoid solid foods initially

**Emergency Warning Signs - Seek immediate care:**
🚨 Blood in vomit or stool
🚨 High fever (>101.5°F/38.6°C)
🚨 Signs of severe dehydration (dizziness, rapid heartbeat, little/no urination)
🚨 Severe abdominal pain
🚨 Unable to keep fluids down for 24+ hours
🚨 Signs of dehydration in children

**Recovery Timeline:**
Most viral GI issues resolve within 24-48 hours with proper care.

**Follow-up Questions:**
• When did symptoms start?
• Any recent travel or new foods?
• Are you able to keep fluids down?
• Any fever or blood in symptoms?

Would you like specific dietary recommendations for recovery?`;
    }

    getMedicationResponse() {
        return `💊 **Medication Guidance & Safety**

I can help with general medication information and safety guidelines:

**What I Can Assist With:**
✅ General medication information and common uses
✅ Typical side effects and what to watch for
✅ Basic dosing guidelines and timing
✅ Storage and safety recommendations
✅ When to contact your healthcare provider

**Medication Safety Reminders:**
⚠️ **Always follow your prescriber's specific instructions**
⚠️ **Never stop medications abruptly** without consulting your doctor
⚠️ **Report severe or unusual side effects** immediately
⚠️ **Keep medications in original containers** with labels
⚠️ **Check expiration dates** regularly
⚠️ **Be aware of drug interactions** - inform doctors of all medications

**Emergency Medication Situations - Call 911:**
🚨 Signs of severe allergic reaction (difficulty breathing, swelling, severe rash)
🚨 Suspected overdose or accidental poisoning
🚨 Severe side effects affecting breathing, consciousness, or heart rate

**Common Questions I Can Help With:**
• "What is this medication used for?"
• "What are common side effects of [medication]?"
• "How should I take this medication?"
• "What should I do if I miss a dose?"
• "Can I take this with other medications?"

**For Specific Help, Please Share:**
• Medication name and strength
• Your specific question or concern
• Any symptoms you're experiencing
• Other medications you're taking

**Important Note:**
For prescription medications, always consult your prescribing physician or pharmacist for personalized advice. This AI provides general educational information only.

What specific medication question can I help you with?`;
    }

    getRespiratoryResponse() {
        return `🫁 **Respiratory Symptom Assessment**

I'll help assess your respiratory symptoms and provide guidance:

**Common Respiratory Conditions:**
• **Common cold** - Runny nose, mild cough, low-grade fever
• **Flu** - High fever, body aches, fatigue, cough
• **COVID-19** - Cough, fever, loss of taste/smell, fatigue
• **Allergies** - Sneezing, runny nose, itchy eyes (no fever)
• **Bronchitis** - Persistent cough with mucus

**Symptom Management:**
✅ **Rest** - Get adequate sleep to help immune system
✅ **Hydration** - Warm fluids (tea, soup, warm water)
✅ **Humidity** - Use humidifier or breathe steam from hot shower
✅ **Throat soothing** - Honey, throat lozenges, salt water gargles
✅ **Cough relief** - Honey (not for children under 1 year)

**COVID-19 Precautions:**
• Consider getting tested if symptoms match COVID-19
• Isolate from others until fever-free for 24 hours
• Wear mask around others if isolation isn't possible
• Monitor for worsening symptoms

**Emergency Warning Signs - Call 911:**
🚨 **Difficulty breathing or shortness of breath**
🚨 **Chest pain or pressure**
🚨 **Bluish lips or face**
🚨 **Severe difficulty swallowing**
🚨 **High fever with breathing problems**

**See a Doctor If:**
• Symptoms worsen after initial improvement
• Fever above 103°F (39.4°C)
• Persistent cough lasting more than 2 weeks
• Coughing up blood
• Severe sinus pressure or headache

**Assessment Questions:**
• When did symptoms start?
• Do you have fever?
• Any difficulty breathing?
• Cough - dry or producing mucus?
• Recent exposure to illness?

Would you like to describe your specific respiratory symptoms in more detail?`;
    }

    getGeneralResponse(userMessage) {
        return `🏥 **Medical Consultation**

Thank you for reaching out. To provide the most helpful guidance, I'd like to learn more about your health concern.

**Please describe:**
• Your main symptoms or health concerns
• When symptoms first appeared
• Severity level (mild, moderate, severe)
• Any factors that make symptoms better or worse
• Your approximate age and general health status
• Current medications or known medical conditions

**I can help you with:**
✅ **Symptom analysis** and preliminary assessment
✅ **Guidance** on when to seek medical care
✅ **Home care** and comfort measures
✅ **Medical information** and health education
✅ **Emergency symptom** recognition

**Medical Disclaimer & Safety:**
⚠️ This AI provides educational information only
⚠️ Always consult healthcare professionals for diagnosis and treatment
⚠️ Seek immediate medical care for emergency symptoms
⚠️ Trust your instincts about your health

**Emergency Symptoms - Call 911 Immediately:**
🚨 Chest pain or pressure
🚨 Difficulty breathing
🚨 Loss of consciousness
🚨 Severe bleeding
🚨 Signs of stroke (face drooping, arm weakness, speech difficulty)
🚨 Severe allergic reactions

**Next Steps:**
Please provide more specific details about your symptoms or health concerns so I can offer more targeted guidance and recommendations.

What specific symptoms or health questions would you like to discuss?`;
    }

    useSuggestion(suggestion) {
        this.elements.messageInput.value = suggestion;
        this.elements.sendBtn.disabled = false;
        this.elements.messageInput.focus();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.elements.chatContainer.scrollTop = this.elements.chatContainer.scrollHeight;
        }, 100);
    }

    showModal(modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    hideModal(modal) {
        modal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }

    showEmergencyModal() {
        this.showModal(this.elements.emergencyModal);
    }

    showSettingsModal() {
        this.showModal(this.elements.settingsModal);
    }

    loadConversation(chatId) {
        console.log('📖 Loading conversation:', chatId);
        
        // Update active state
        document.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
        });
        
        document.querySelector(`[data-chat-id="${chatId}"]`)?.classList.add('active');
        
        // For demo purposes, we'll just start a new conversation
        // In a real app, you'd load the conversation from storage
        this.startNewConversation();
    }

    loadConversations() {
        // In a real app, this would load conversations from local storage or a server
        console.log('📚 Loading conversation history...');
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Medical AI Chat Application...');
    window.medicalChatApp = new MedicalChatApp();
});

// Service worker registration for PWA functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('✅ SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('❌ SW registration failed: ', registrationError);
            });
    });
}

// Handle offline functionality
window.addEventListener('online', () => {
    console.log('🌐 Connection restored');
});

window.addEventListener('offline', () => {
    console.log('📴 Connection lost - working offline');
});

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MedicalChatApp;
}