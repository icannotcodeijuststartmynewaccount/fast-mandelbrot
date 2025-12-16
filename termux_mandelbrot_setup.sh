#!/data/data/com.termux/files/usr/bin/bash
echo "=== Termux Setup ==="
pkg update && pkg upgrade
pkg install python python-numpy clang
pip install -r requirements_termux.txt
echo "✅ Termux setup complete!"