#!/bin/bash

# Git Commit History Rewriter
# This script will rewrite commit messages to follow uniform convention

set -e

# Use current directory instead of hardcoded path
PROJECT_DIR="$(pwd)"
echo "=== Git Commit History Rewriter ==="
echo "Current directory: $PROJECT_DIR"
echo ""

# Create backup branch
echo "Step 1: Creating backup branch..."
git branch backup-before-rewrite 2>/dev/null || echo "Backup branch already exists"
echo "Backup created: backup-before-rewrite"
echo ""

# Create the commit message mapping file
echo "Step 2: Creating commit message mapping..."
cat > /tmp/commit-messages.txt << 'COMMITS'
feat: Initialize Multi-Model Orchestration project scaffold
feat(phase 1.1): Add automated data ingestion with validation and checksum verification
feat(phase 1.1): Add multi-source dataset merger (4,786 unique samples)
feat(phase 1.2): Add feature engineering pipeline with text cleaning and label encoding
feat(phase 1.3): Add stratified data splitting (4,774 samples, 41 intents)
feat(phase 1.4): Add data versioning and lineage tracking (VERSION.md + checksums)
feat(phase 1.5): Rename scripts to Phase 1 convention (p1.1, p1.2, p1.3)
feat(phase 2.1): Add document preprocessing pipeline
feat(phase 2.2): Add text embedding generation with TF-IDF (5K features, 91MB)
chore: Update .gitignore to exclude embeddings and input-data files
feat(phase 2.3): Add embedding validation with cohesion/separation metrics
feat(phase 2.4): Add embedding storage with versioning and checksums (v2.0)
feat(phase 3.1): Add vector database setup with FAISS index (4774 vectors, 91MB)
chore: Update .gitignore to exclude FAISS index binary files
feat(phase 3.2 & 3.3): Implement FAISS similarity search with k-NN retrieval and evaluation metrics
COMMITS

echo "Commit messages prepared."
echo ""

# Create the rebase-todo file
echo "Step 3: Preparing interactive rebase..."
git rev-list --reverse HEAD | head -n 15 > /tmp/commit-hashes.txt

# Create rebase-todo
cat > /tmp/rebase-todo.txt << 'REBASE'
REBASE

counter=1
while read -r hash; do
    if [ $counter -eq 1 ]; then
        echo "pick $hash" >> /tmp/rebase-todo.txt
    else
        echo "reword $hash" >> /tmp/rebase-todo.txt
    fi
    ((counter++))
done < /tmp/commit-hashes.txt

echo "Rebase todo created with $(wc -l < /tmp/rebase-todo.txt) commits"
echo ""

# Create editor script for git
cat > /tmp/git-editor.sh << 'EDITOR'
#!/bin/bash
# Git editor script for rebase

if [[ "$GIT_SEQUENCE_EDITOR" == *"rebase"* ]] || [[ "$1" == *"rebase-todo"* ]]; then
    # This is the rebase-todo file
    cp /tmp/rebase-todo.txt "$1"
else
    # This is a commit message file
    commit_num=$(echo "$1" | grep -o '[0-9]\+' | head -1)
    if [ -n "$commit_num" ] && [ "$commit_num" -le 15 ]; then
        sed -n "${commit_num}p" /tmp/commit-messages.txt > "$1"
    fi
fi
EDITOR

chmod +x /tmp/git-editor.sh

echo "Step 4: Starting interactive rebase..."
echo "This will rewrite all commit messages..."
echo ""

# Set the editor and start rebase
export GIT_SEQUENCE_EDITOR="/tmp/git-editor.sh"
export EDITOR="/tmp/git-editor.sh"

git rebase -i --root

echo ""
echo "=== Rewrite Complete! ==="
echo "View your new history with: git log --oneline"
echo "To undo and restore backup: git reset --hard backup-before-rewrite"
echo ""

# Cleanup
rm -f /tmp/commit-messages.txt /tmp/commit-hashes.txt /tmp/rebase-todo.txt /tmp/git-editor.sh

echo "Script completed successfully!"
