# MindPulse – Backend (Emotion Analysis API)

## Overview
MindPulse is an AI-powered system that performs real-time emotion analysis using facial input.  
This repository contains the backend API responsible for processing input, running the model, and returning predictions.

## Tech Stack
- FastAPI  
- TensorFlow  
- OpenCV  
- Pandas  

## Features
- Real-time facial emotion detection  
- REST API for model inference  
- Data logging for analysis  
- Integration-ready with frontend dashboard  

## Workflow
1. User input (image/video frame) is sent to backend API  
2. Image is processed using OpenCV  
3. Model predicts emotion using trained deep learning model  
4. Result is returned and optionally logged for analysis  

## API Endpoint Example
POST `/predict`  
Input: Image frame  
Output: Emotion label + confidence  

## Project Structure
- `api.py` → API routes  
- `model/` → trained model files   

## Note
This is the backend service. Frontend is available in a separate repository.
