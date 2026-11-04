import { ChatGoogle } from "@langchain/google";
import { ChatMistralAI } from "@langchain/mistralai"; 
import { ChatGroq } from "@langchain/groq";
import config from "../config/config.js";

export const geminiModel = new ChatGoogle({
  model: "gemini-3.5-flash",
  apiKey: config.GEMINI_API_KEY,
});

export const mistralModel = new ChatMistralAI({
  model: "ministral-8b-latest",
  apiKey: config.MISTRAL_API_KEY,
});

export const groqModel = new ChatGroq({
  apiKey: config.GROQ_API_KEY,
  model: "openai/gpt-oss-20b",
  temperature: 0,
});
