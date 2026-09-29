# Tasks: Enterprise Agentic RAG Azure Deployment

## Phase 1: Environment & Pre-deployment Verification
- [x] 1.1 Verify Azure login status and active subscription (Azure for Students) <!-- id: 0 -->
- [x] 1.2 Validate project structure, Dockerfile, docker-compose.yml, and environment variable requirements <!-- id: 1 -->
- [x] 1.3 Collect and configure `.env` secrets (`GROQ_API_KEY`, `GOOGLEGEMINI_API_KEY`, `QDRANT_API_KEY`, `PORTKEY_API_KEY`, `AUTH_SECRET_KEY`) <!-- id: 2 -->

## Phase 2: Azure Infrastructure Provisioning (Azure CLI)
- [x] 2.1 Create Azure Resource Group `rg-agentic-rag-prod` in `centralindia` <!-- id: 3 -->
- [x] 2.2 Provision Azure Virtual Machine (`Standard_B2s_v2`, Ubuntu 24.04 LTS) <!-- id: 4 -->
- [x] 2.3 Configure Network Security Group (NSG) to allow ports 8501 (UI), 8000 (Backend API), and 22 (SSH) <!-- id: 5 -->

## Phase 3: Containerization & Stack Deployment
- [x] 3.1 Provision Docker & Docker Compose on Azure VM <!-- id: 6 -->
- [x] 3.2 Deploy application code and `.env` configuration to Azure VM <!-- id: 7 -->
- [x] 3.3 Build Docker images (`docker compose build`) <!-- id: 8 -->
- [x] 3.4 Execute initial data ingestion into Qdrant Cloud (`docker compose run --rm backend ...`) <!-- id: 9 -->
- [x] 3.5 Launch full multi-container stack (`docker compose up -d`) <!-- id: 10 -->

## Phase 4: Post-Deployment Verification & Resource Management
- [x] 4.1 Verify FastAPI backend health (`http://20.219.21.38:8000/`) -> Returned HTTP 200 OK <!-- id: 11 -->
- [x] 4.2 Verify Streamlit UI functionality (`http://20.219.21.38:8501/`) -> Returned HTTP 200 OK <!-- id: 12 -->
- [x] 4.3 Create helper management scripts `on.sh` and `off.sh` for resource efficiency <!-- id: 13 -->
- [x] 4.4 Create final Walkthrough artifact with access endpoints and management instructions <!-- id: 14 -->
