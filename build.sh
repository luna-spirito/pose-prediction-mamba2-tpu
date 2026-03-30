SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
stickytape --add-python-path "$PARENT_DIR" "$SCRIPT_DIR/modules/main.py" --output-file "$SCRIPT_DIR/result.py"
