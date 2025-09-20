import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StatusBar,
  SafeAreaView,
  KeyboardAvoidingView,
  Platform,
  Alert,
  Animated,
  Dimensions,
  Modal
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { Ionicons, MaterialIcons } from '@expo/vector-icons';

// Get screen dimensions
const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// ChatGPT-style color palette
const colors = {
  primary: '#212121',
  secondary: '#171717',
  tertiary: '#2f2f2f',
  quaternary: '#1a1a1a',
  textPrimary: '#ececec',
  textSecondary: '#b4b4b4',
  textTertiary: '#8e8ea0',
  accent: '#10a37f',
  accentHover: '#0d8f6f',
  accentLight: 'rgba(16, 163, 127, 0.1)',
  border: '#4d4d4f',
  borderLight: '#565869',
  danger: '#ff4757',
  warning: '#ffa502',
  success: '#2ed573',
};

// Message component matching ChatGPT style
const ChatMessage = ({ message, isUser, isTyping = false }) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const formatMessage = (text) => {
    // Simple formatting for mobile
    return text.replace(/\*\*(.*?)\*\*/g, '$1').replace(/\*(.*?)\*/g, '$1');
  };

  if (isTyping) {
    return (
      <Animated.View
        style={[
          styles.messageContainer,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }
        ]}
      >
        <View style={styles.assistantAvatar}>
          <Text style={styles.avatarText}>AI</Text>
        </View>
        <View style={styles.typingBubble}>
          <TypingIndicator />
        </View>
      </Animated.View>
    );
  }

  return (
    <Animated.View
      style={[
        styles.messageContainer,
        isUser && styles.userMessageContainer,
        { opacity: fadeAnim, transform: [{ translateY: slideAnim }] }
      ]}
    >
      <View style={isUser ? styles.userAvatar : styles.assistantAvatar}>
        <Text style={styles.avatarText}>{isUser ? 'U' : 'AI'}</Text>
      </View>
      <View style={isUser ? styles.userMessage : styles.assistantMessage}>
        <Text style={isUser ? styles.userMessageText : styles.assistantMessageText}>
          {formatMessage(message)}
        </Text>
      </View>
    </Animated.View>
  );
};

// Typing indicator component
const TypingIndicator = () => {
  const dot1 = useRef(new Animated.Value(0.3)).current;
  const dot2 = useRef(new Animated.Value(0.3)).current;
  const dot3 = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    const animate = () => {
      Animated.sequence([
        Animated.timing(dot1, { toValue: 1, duration: 400, useNativeDriver: true }),
        Animated.timing(dot1, { toValue: 0.3, duration: 400, useNativeDriver: true }),
      ]).start();

      setTimeout(() => {
        Animated.sequence([
          Animated.timing(dot2, { toValue: 1, duration: 400, useNativeDriver: true }),
          Animated.timing(dot2, { toValue: 0.3, duration: 400, useNativeDriver: true }),
        ]).start();
      }, 200);

      setTimeout(() => {
        Animated.sequence([
          Animated.timing(dot3, { toValue: 1, duration: 400, useNativeDriver: true }),
          Animated.timing(dot3, { toValue: 0.3, duration: 400, useNativeDriver: true }),
        ]).start();
      }, 400);
    };

    animate();
    const interval = setInterval(animate, 1400);
    return () => clearInterval(interval);
  }, []);

  return (
    <View style={styles.typingContainer}>
      <Animated.View style={[styles.typingDot, { opacity: dot1 }]} />
      <Animated.View style={[styles.typingDot, { opacity: dot2 }]} />
      <Animated.View style={[styles.typingDot, { opacity: dot3 }]} />
    </View>
  );
};

// Suggestion button component
const SuggestionButton = ({ title, onPress, icon }) => {
  return (
    <TouchableOpacity 
      style={styles.suggestionButton} 
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
    >
      <Text style={styles.suggestionIcon}>{icon}</Text>
      <Text style={styles.suggestionText}>{title}</Text>
    </TouchableOpacity>
  );
};

// Emergency modal component
const EmergencyModal = ({ visible, onClose }) => {
  return (
    <Modal
      animationType="slide"
      transparent={true}
      visible={visible}
      onRequestClose={onClose}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.emergencyModal}>
          <View style={styles.emergencyHeader}>
            <Text style={styles.emergencyIcon}>🚨</Text>
            <Text style={styles.emergencyTitle}>Medical Emergency</Text>
            <TouchableOpacity onPress={onClose} style={styles.modalCloseBtn}>
              <Ionicons name="close" size={24} color={colors.textTertiary} />
            </TouchableOpacity>
          </View>
          
          <ScrollView style={styles.emergencyContent}>
            <View style={styles.emergencyWarning}>
              <Text style={styles.emergencyWarningText}>
                <Text style={styles.boldText}>If this is a medical emergency, do not rely on this AI.</Text>
                {'\n\n'}Call emergency services immediately:
              </Text>
            </View>
            
            <View style={styles.emergencyContacts}>
              <TouchableOpacity style={styles.emergencyContact}>
                <Text style={styles.contactIcon}>🚑</Text>
                <View style={styles.contactInfo}>
                  <Text style={styles.contactTitle}>Emergency Services</Text>
                  <Text style={styles.contactNumber}>911</Text>
                </View>
                <TouchableOpacity style={styles.callButton}>
                  <Text style={styles.callButtonText}>Call</Text>
                </TouchableOpacity>
              </TouchableOpacity>
              
              <TouchableOpacity style={styles.emergencyContact}>
                <Text style={styles.contactIcon}>☠️</Text>
                <View style={styles.contactInfo}>
                  <Text style={styles.contactTitle}>Poison Control</Text>
                  <Text style={styles.contactNumber}>1-800-222-1222</Text>
                </View>
                <TouchableOpacity style={styles.callButton}>
                  <Text style={styles.callButtonText}>Call</Text>
                </TouchableOpacity>
              </TouchableOpacity>
            </View>
            
            <View style={styles.emergencySigns}>
              <Text style={styles.emergencySignsTitle}>Emergency Warning Signs:</Text>
              <Text style={styles.emergencySignsList}>
                • Chest pain or pressure{'\n'}
                • Difficulty breathing{'\n'}
                • Loss of consciousness{'\n'}
                • Severe bleeding{'\n'}
                • Signs of stroke{'\n'}
                • Severe allergic reactions
              </Text>
            </View>
          </ScrollView>
        </View>
      </View>
    </Modal>
  );
};

// Main App Component
export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [emergencyModalVisible, setEmergencyModalVisible] = useState(false);
  const scrollViewRef = useRef();
  const inputRef = useRef();

  // Initialize with welcome message
  useEffect(() => {
    setMessages([{
      id: '1',
      text: `👋 Welcome to your Medical AI Assistant!

I'm here to help with your health questions and concerns. I can assist with:

• Symptom analysis and preliminary assessment
• Medical questions and health guidance  
• Emergency symptom detection
• Medication information
• When to seek professional medical care

Please describe your symptoms, ask your medical questions, or use the suggested prompts below.

⚠️ **Important Medical Disclaimer:**
This AI provides educational information only. Always consult qualified healthcare professionals for medical diagnosis, treatment, and emergency care. In case of emergency, call 911 immediately.`,
      isUser: false,
      timestamp: Date.now()
    }]);
  }, []);

  // Medical AI response generator
  const getMedicalResponse = async (userMessage) => {
    const messageLower = userMessage.toLowerCase();
    
    // Emergency detection
    const emergencyKeywords = [
      'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 
      'difficulty breathing', 'severe bleeding', 'unconscious',
      'severe allergic reaction', 'overdose', 'poisoning'
    ];
    
    if (emergencyKeywords.some(keyword => messageLower.includes(keyword))) {
      return getEmergencyResponse();
    }
    
    // Fever symptoms
    if (messageLower.includes('fever') || messageLower.includes('temperature')) {
      return getFeverResponse();
    }
    
    // Digestive issues
    if (messageLower.includes('stomach') || messageLower.includes('nausea') || messageLower.includes('vomit')) {
      return getDigestiveResponse();
    }
    
    // Medication questions
    if (messageLower.includes('medication') || messageLower.includes('medicine') || messageLower.includes('drug')) {
      return getMedicationResponse();
    }
    
    // General response
    return getGeneralResponse();
  };

  const getEmergencyResponse = () => {
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

⚠️ **This is an AI assessment. If you believe this is an emergency, call 911 immediately regardless of this analysis.**

Time is critical in medical emergencies. Professional medical care is essential.`;
  };

  const getFeverResponse = () => {
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
🚨 Severe dehydration
🚨 Persistent vomiting

Would you like to describe any additional symptoms like headache, body aches, or respiratory symptoms?`;
  };

  const getDigestiveResponse = () => {
    return `🤢 **Digestive System Analysis**

Your symptoms suggest gastrointestinal concerns:

**Possible Causes:**
• Viral gastroenteritis (stomach flu)
• Food poisoning or foodborne illness
• Dietary indiscretion
• Medication side effects

**Immediate Treatment:**
✅ Clear fluids (water, electrolyte solutions)
✅ BRAT diet: Bananas, Rice, Applesauce, Toast
✅ Small, frequent sips rather than large amounts
✅ Avoid dairy, caffeine, alcohol, fatty foods

**Warning Signs - Seek immediate care:**
🚨 Blood in vomit or stool
🚨 High fever (>101.5°F)
🚨 Signs of severe dehydration
🚨 Unable to keep fluids down for 24+ hours

Most viral GI issues resolve within 24-48 hours with proper care.`;
  };

  const getMedicationResponse = () => {
    return `💊 **Medication Assistance**

I can help with various medication-related questions:

**What I can assist with:**
• General medication information
• Common side effects and interactions
• Proper dosing and timing guidelines
• Storage and safety recommendations

**Safety Reminders:**
⚠️ Always follow your prescriber's instructions
⚠️ Don't stop medications abruptly without consulting your doctor
⚠️ Report unusual or severe side effects immediately
⚠️ Keep medications in original labeled containers

**Emergency Situations:**
If experiencing severe allergic reactions or signs of overdose, call 911 immediately.

What specific medication question can I help you with today?`;
  };

  const getGeneralResponse = () => {
    return `🏥 **Medical Consultation**

To provide the most helpful guidance, please tell me about:

• Your main symptoms or health concerns
• When symptoms first appeared
• Severity level (mild, moderate, severe)
• Any factors that make symptoms better or worse
• Current medications or known medical conditions

**I can help you with:**
✅ Symptom analysis and preliminary assessment
✅ Guidance on when to seek medical care
✅ Home care and comfort measures
✅ Medical information and health education
✅ Emergency symptom recognition

Please provide more details about your specific symptoms or concerns so I can offer more targeted guidance.`;
  };

  // Send message function
  const sendMessage = async () => {
    if (!inputText.trim() || isTyping) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    const userMessage = {
      id: Date.now().toString(),
      text: inputText.trim(),
      isUser: true,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setShowSuggestions(false);
    setIsTyping(true);
    
    // Check for emergency keywords
    const emergencyKeywords = ['chest pain', 'heart attack', 'stroke', 'emergency', 'help me', 'dying'];
    if (emergencyKeywords.some(keyword => inputText.toLowerCase().includes(keyword))) {
      setTimeout(() => setEmergencyModalVisible(true), 1000);
    }

    // Simulate AI processing delay
    setTimeout(async () => {
      const aiResponse = await getMedicalResponse(inputText);
      
      const responseMessage = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        isUser: false,
        timestamp: Date.now()
      };

      setMessages(prev => [...prev, responseMessage]);
      setIsTyping(false);
    }, 2000 + Math.random() * 1000);
  };

  // Suggestion handler
  const useSuggestion = (suggestion) => {
    setInputText(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  // New chat handler
  const startNewChat = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setMessages([]);
    setInputText('');
    setShowSuggestions(true);
    setIsTyping(false);
  };

  const suggestions = [
    { title: 'Fever symptoms', icon: '🤒', text: 'I\'m experiencing fever and headache symptoms' },
    { title: 'Chest pain concern', icon: '💓', text: 'I have chest pain and need guidance' },
    { title: 'Medication question', icon: '💊', text: 'I have questions about my medication' },
    { title: 'Digestive problems', icon: '🤢', text: 'I\'m having digestive issues' },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={colors.primary} />
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={startNewChat} style={styles.newChatButton}>
          <Ionicons name="add" size={20} color={colors.textPrimary} />
          <Text style={styles.newChatText}>New consultation</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          onPress={() => setEmergencyModalVisible(true)}
          style={styles.emergencyButton}
        >
          <Text style={styles.emergencyButtonText}>🚨</Text>
        </TouchableOpacity>
      </View>

      {/* Chat Area */}
      <KeyboardAvoidingView 
        style={styles.chatContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={false}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map(message => (
            <ChatMessage
              key={message.id}
              message={message.text}
              isUser={message.isUser}
            />
          ))}
          
          {isTyping && <ChatMessage isTyping={true} />}
        </ScrollView>

        {/* Input Area */}
        <View style={styles.inputArea}>
          {showSuggestions && (
            <ScrollView
              horizontal
              style={styles.suggestionsContainer}
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.suggestionsContent}
            >
              {suggestions.map((suggestion, index) => (
                <SuggestionButton
                  key={index}
                  title={suggestion.title}
                  icon={suggestion.icon}
                  onPress={() => useSuggestion(suggestion.text)}
                />
              ))}
            </ScrollView>
          )}
          
          <View style={styles.inputContainer}>
            <TextInput
              ref={inputRef}
              style={styles.textInput}
              value={inputText}
              onChangeText={setInputText}
              placeholder="Describe your symptoms or ask a medical question..."
              placeholderTextColor={colors.textTertiary}
              multiline
              maxLength={2000}
              returnKeyType="send"
              onSubmitEditing={sendMessage}
            />
            
            <TouchableOpacity
              style={[
                styles.sendButton,
                { opacity: inputText.trim() ? 1 : 0.5 }
              ]}
              onPress={sendMessage}
              disabled={!inputText.trim() || isTyping}
            >
              <Ionicons 
                name="send" 
                size={20} 
                color="white" 
              />
            </TouchableOpacity>
          </View>
          
          <Text style={styles.disclaimer}>
            Medical AI Assistant can make mistakes. Consider checking important medical information with healthcare professionals.
          </Text>
        </View>
      </KeyboardAvoidingView>

      {/* Emergency Modal */}
      <EmergencyModal
        visible={emergencyModalVisible}
        onClose={() => setEmergencyModalVisible(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.secondary,
  },
  newChatButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  newChatText: {
    color: colors.textPrimary,
    fontSize: 14,
    fontWeight: '500',
    marginLeft: 6,
  },
  emergencyButton: {
    width: 40,
    height: 40,
    backgroundColor: colors.danger,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emergencyButtonText: {
    fontSize: 20,
  },
  chatContainer: {
    flex: 1,
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  messageContainer: {
    flexDirection: 'row',
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  userMessageContainer: {
    flexDirection: 'row-reverse',
  },
  assistantAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.tertiary,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  userAvatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: colors.textPrimary,
    fontSize: 12,
    fontWeight: '600',
  },
  assistantMessage: {
    flex: 1,
    marginLeft: 12,
    padding: 0,
  },
  userMessage: {
    backgroundColor: colors.accent,
    marginRight: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 18,
    borderTopRightRadius: 4,
    maxWidth: screenWidth * 0.8,
  },
  assistantMessageText: {
    color: colors.textPrimary,
    fontSize: 16,
    lineHeight: 24,
  },
  userMessageText: {
    color: 'white',
    fontSize: 16,
    lineHeight: 24,
  },
  typingBubble: {
    backgroundColor: colors.secondary,
    borderWidth: 1,
    borderColor: colors.border,
    marginLeft: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 18,
  },
  typingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  typingDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.textTertiary,
    marginHorizontal: 2,
  },
  inputArea: {
    backgroundColor: colors.primary,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 16,
  },
  suggestionsContainer: {
    marginBottom: 12,
  },
  suggestionsContent: {
    paddingHorizontal: 0,
  },
  suggestionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.secondary,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginRight: 8,
  },
  suggestionIcon: {
    marginRight: 6,
    fontSize: 14,
  },
  suggestionText: {
    color: colors.textSecondary,
    fontSize: 14,
  },
  inputContainer: {
    backgroundColor: colors.secondary,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 4,
    paddingVertical: 4,
    marginBottom: 8,
  },
  textInput: {
    flex: 1,
    color: colors.textPrimary,
    fontSize: 16,
    paddingHorizontal: 12,
    paddingVertical: 8,
    maxHeight: 120,
    minHeight: 40,
  },
  sendButton: {
    width: 36,
    height: 36,
    backgroundColor: colors.accent,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    margin: 2,
  },
  disclaimer: {
    textAlign: 'center',
    color: colors.textTertiary,
    fontSize: 12,
    lineHeight: 16,
  },
  // Emergency Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  emergencyModal: {
    backgroundColor: colors.secondary,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 16,
    width: screenWidth * 0.9,
    maxHeight: screenHeight * 0.8,
  },
  emergencyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  emergencyIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  emergencyTitle: {
    flex: 1,
    color: colors.textPrimary,
    fontSize: 20,
    fontWeight: '600',
  },
  modalCloseBtn: {
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emergencyContent: {
    padding: 20,
  },
  emergencyWarning: {
    backgroundColor: 'rgba(255, 71, 87, 0.1)',
    borderWidth: 1,
    borderColor: colors.danger,
    borderRadius: 8,
    padding: 16,
    marginBottom: 20,
  },
  emergencyWarningText: {
    color: colors.textPrimary,
    fontSize: 14,
    lineHeight: 20,
  },
  boldText: {
    fontWeight: 'bold',
  },
  emergencyContacts: {
    marginBottom: 20,
  },
  emergencyContact: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.tertiary,
    borderRadius: 8,
    padding: 16,
    marginBottom: 8,
  },
  contactIcon: {
    fontSize: 20,
    width: 40,
    textAlign: 'center',
  },
  contactInfo: {
    flex: 1,
    marginLeft: 12,
  },
  contactTitle: {
    color: colors.textPrimary,
    fontSize: 16,
    fontWeight: '600',
  },
  contactNumber: {
    color: colors.accent,
    fontSize: 18,
    fontWeight: '700',
  },
  callButton: {
    backgroundColor: colors.accent,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  callButtonText: {
    color: 'white',
    fontWeight: '600',
  },
  emergencySigns: {
    marginTop: 16,
  },
  emergencySignsTitle: {
    color: colors.textPrimary,
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  emergencySignsList: {
    color: colors.textSecondary,
    fontSize: 14,
    lineHeight: 20,
  },
});