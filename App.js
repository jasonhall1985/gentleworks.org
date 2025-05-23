import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView, ScrollView } from 'react-native';

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  const suggestedPhrases = [
    'I need help',
    "I'm in pain",
    "I'm thirsty",
    'Thank you'
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* Video Preview Placeholder */}
        <View style={styles.videoPreview}>
          <Text style={styles.videoText}>Video Preview</Text>
        </View>
        {/* Record Button */}
        <TouchableOpacity
          style={[styles.recordButton, isRecording ? styles.recording : styles.idle]}
          onPress={() => setIsRecording(!isRecording)}
          accessibilityLabel={isRecording ? 'Stop recording' : 'Start recording'}
        >
          <Text style={styles.recordIcon}>{isRecording ? '■' : '●'}</Text>
        </TouchableOpacity>
        {/* Recognized Text */}
        <View style={styles.textOutput}>
          <Text style={styles.textOutputText}>
            {recognizedText || 'Recognition result will appear here'}
          </Text>
        </View>
        {/* Suggested Phrases */}
        <View style={styles.suggestedContainer}>
          {suggestedPhrases.map((phrase) => (
            <TouchableOpacity
              key={phrase}
              style={styles.suggestedButton}
              onPress={() => setRecognizedText(phrase)}
              accessibilityLabel={phrase}
            >
              <Text style={styles.suggestedText}>{phrase}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContainer: {
    alignItems: 'center',
    paddingVertical: 32,
    paddingHorizontal: 20,
  },
  videoPreview: {
    width: '100%',
    height: 260,
    backgroundColor: '#e0e0e0',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 32,
  },
  videoText: {
    color: '#888',
    fontSize: 20,
  },
  recordButton: {
    width: 90,
    height: 90,
    borderRadius: 45,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 28,
  },
  recording: {
    backgroundColor: '#e53935',
  },
  idle: {
    backgroundColor: '#1976d2',
  },
  recordIcon: {
    color: '#fff',
    fontSize: 40,
  },
  textOutput: {
    width: '100%',
    minHeight: 60,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
    padding: 10,
  },
  textOutputText: {
    fontSize: 20,
    color: '#222',
    textAlign: 'center',
  },
  suggestedContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 10,
  },
  suggestedButton: {
    backgroundColor: '#e3e3e3',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    margin: 5,
  },
  suggestedText: {
    fontSize: 16,
    color: '#222',
  },
});
