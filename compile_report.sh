#!/bin/bash
#
# compile_report.sh - Compile LaTeX Report to PDF
# =================================================
#
# This script compiles the generated LaTeX report to PDF.
# Installs texlive if needed.

set -e

REPORT_FILE="final_report.tex"
PDF_FILE="final_report.pdf"

echo "============================================================"
echo "LaTeX Report Compilation"
echo "============================================================"

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Installing texlive..."
    
    # Detect OS and install
    if [ -f /etc/arch-release ]; then
        echo "Detected Arch Linux, installing texlive-core..."
        sudo pacman -S --noconfirm texlive-core texlive-bin texlive-latexextra
    elif [ -f /etc/debian_version ]; then
        echo "Detected Debian/Ubuntu, installing texlive..."
        sudo apt-get update
        sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
    elif [ -f /etc/fedora-release ]; then
        echo "Detected Fedora/RHEL, installing texlive..."
        sudo dnf install -y texlive-scheme-basic texlive-latex
    else
        echo "❌ Unknown OS. Please install texlive manually:"
        echo "   Arch: sudo pacman -S texlive-core texlive-bin"
        echo "   Ubuntu: sudo apt-get install texlive-latex-base texlive-latex-extra"
        echo "   Fedora: sudo dnf install texlive-scheme-basic"
        exit 1
    fi
fi

echo ""
echo "✓ pdflatex found: $(which pdflatex)"

# Check if report exists
if [ ! -f "$REPORT_FILE" ]; then
    echo "❌ Report file not found: $REPORT_FILE"
    echo "   Run: bash run_complete_analysis.sh"
    exit 1
fi

echo ""
echo "📄 Compiling LaTeX report..."
echo "   Input: $REPORT_FILE"
echo "   Output: $PDF_FILE"

# Compile (run twice for TOC and references)
echo ""
echo "[1/2] First pass..."
pdflatex -interaction=nonstopmode "$REPORT_FILE" > compile_log1.txt 2>&1 || {
    echo "❌ First pass failed. Check compile_log1.txt for errors."
    tail -50 compile_log1.txt
    exit 1
}

echo "[2/2] Second pass (for TOC)..."
pdflatex -interaction=nonstopmode "$REPORT_FILE" > compile_log2.txt 2>&1 || {
    echo "❌ Second pass failed. Check compile_log2.txt for errors."
    tail -50 compile_log2.txt
    exit 1
}

# Check if PDF was created
if [ -f "$PDF_FILE" ]; then
    PDF_SIZE=$(ls -lh "$PDF_FILE" | awk '{print $5}')
    echo ""
    echo "============================================================"
    echo "✅ Compilation successful!"
    echo "============================================================"
    echo ""
    echo "📊 Generated: $PDF_FILE ($PDF_SIZE)"
    echo ""
    echo "To view:"
    echo "   xdg-open $PDF_FILE"
    echo "   evince $PDF_FILE"
    echo "   okular $PDF_FILE"
    echo ""
    
    # Clean up auxiliary files
    echo "Cleaning up auxiliary files..."
    rm -f *.aux *.log *.toc *.out compile_log*.txt
    echo "✓ Done!"
else
    echo "❌ PDF not created. Check compile_log*.txt for errors."
    exit 1
fi
