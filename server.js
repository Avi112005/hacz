const express = require('express');
const multer = require('multer');
const fs = require('fs');
const cors = require('cors');
const ffmpeg = require('fluent-ffmpeg');
const Groq = require('groq-sdk');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');

dotenv.config();

const app = express();
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// === /api/chat — Groq chat with coder model switch ===
app.post('/api/chat', async (req, res) => {
  const { message, language } = req.body;

  const isCodingQuery = /code|function|loop|program|syntax|bug|error|compile|algorithm|write|java|python|c\+\+|html|css|javascript|react/i.test(message);

  const model = isCodingQuery
    ? "qwen-2.5-coder-32b"
    : "meta-llama/llama-4-scout-17b-16e-instruct";

  try {
    const chatCompletion = await groq.chat.completions.create({
      model,
      messages: [
        {
          role: "system",
          content: `You are a helpful ${isCodingQuery ? "AI coding assistant" : "multilingual AI assistant"}. Respond in this language: ${language}.`
        },
        { role: "user", content: message }
      ],
      temperature: isCodingQuery ? 0.6 : 1,
      max_completion_tokens: isCodingQuery ? 8192 : 1024,
      top_p: 0.95
    });

    const reply = chatCompletion.choices[0].message.content;
    res.json({ reply });
  } catch (error) {
    console.error("Groq Chat API error:", error);
    res.status(500).json({ error: 'Chat failed' });
  }
});

// === /api/vision — Gemini 1.5 Flash Image-to-Text ===
app.post('/api/vision', async (req, res) => {
  const { base64Image, message } = req.body;
  if (!base64Image) return res.status(400).json({ error: 'No image provided.' });

  try {
    const base64Data = base64Image.split(',')[1];
    const mimeType = base64Image.match(/^data:(image\/[a-z]+);base64/)?.[1] || 'image/jpeg';

    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const result = await model.generateContent({
      contents: [
        {
          role: 'user',
          parts: [
            { text: message || 'Describe this image.' },
            {
              inlineData: {
                data: base64Data,
                mimeType
              }
            }
          ]
        }
      ]
    });

    const reply = await result.response.text();
    res.json({ reply });
  } catch (err) {
    console.error("Gemini Vision API error:", err.message || err);
    res.status(500).json({ error: 'Gemini Vision processing failed' });
  }
});

// === /api/stt — Speech-to-Text via Groq (Whisper) with FFmpeg Conversion ===
app.post('/api/stt', upload.single('audio'), async (req, res) => {
  const audioFilePath = req.file.path;
  const convertedFilePath = `${audioFilePath}.webm`; // Convert to a supported format

  try {
    // Convert the audio file using FFmpeg
    await new Promise((resolve, reject) => {
      ffmpeg(audioFilePath)
        .toFormat('webm')
        .on('end', resolve)
        .on('error', reject)
        .save(convertedFilePath);
    });

    // Upload the converted file to Groq STT API
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(convertedFilePath),
      model: "whisper-large-v3",
      response_format: "verbose_json"
    });

    res.json({ text: transcription.text });
  } catch (err) {
    console.error("STT error:", err);
    res.status(500).json({ error: 'Transcription failed' });
  } finally {
    // Cleanup original and converted files
    fs.unlinkSync(audioFilePath);
    fs.unlinkSync(convertedFilePath);
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`GroqMate API running at http://localhost:${PORT}`);
});
