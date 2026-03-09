# Multi-Model Orchestration System

## Overview
Production-grade model routing and orchestration system implementing intelligent request distribution across multiple ML models (RAG, Fraud Detection, Classification). Built on constrained hardware (8GB RAM, NVIDIA MX150 2GB) to demonstrate FinOps-first MLOps engineering.

## Business Problem
Enterprises deploy multiple ML models but lack intelligent routing mechanisms to:
- Direct requests to appropriate models based on intent
- Manage model versions and canary deployments
- Optimize resource utilization under hardware constraints
- Maintain production-grade observability

## Solution
A lightweight orchestration layer that:
- Classifies incoming request intent in real-time
- Routes to optimal model endpoint (RAG, Fraud, or Fallback)
- Implements canary deployment with automatic rollback
- Operates within 8GB RAM constraint (FinOps-aware)

## Technical Stack
- **Language**: Python 3.12.3
- **Orchestration**: Custom router with intent classification
- **Models**: RAG (ChromaDB), Fraud Detection (XGBoost), Intent Classifier (sklearn)
- **Deployment**: Docker multi-stage builds, GitHub Container Registry
- **CI/CD**: GitHub Actions with security gating
- **Hardware**: Intel i5-8265U, 8GB RAM, NVIDIA MX150 2GB VRAM

## Project Structure
