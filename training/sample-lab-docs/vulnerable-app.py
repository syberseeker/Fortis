"""
Sample Python Web Application — Workshop Lab Document
Intentionally contains security vulnerabilities for analysis.
This represents a simplified user management API.
"""

import os
import sqlite3
import pickle
import hashlib
import subprocess
import yaml
import tempfile

from flask import Flask, request, jsonify

app = Flask(__name__)

# Hardcoded database credentials
DB_HOST = "prod-db.company.internal"
DB_USER = "admin"
DB_PASS = "Sup3rS3cret!2024"
API_KEY = "sk-1234567890abcdef1234567890abcdef"

# Database connection — no connection pooling
def get_db():
    conn = sqlite3.connect('users.db')
    return conn

# Insecure password hashing — using MD5
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

# User registration — no input validation
@app.route('/api/users', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    conn = get_db()
    cursor = conn.cursor()
    
    # SQL injection vulnerability
    query = f"INSERT INTO users (username, password, email) VALUES ('{username}', '{hash_password(password)}', '{email}')"
    cursor.execute(query)
    conn.commit()
    
    return jsonify({"status": "registered", "username": username}), 201

# Login endpoint — no rate limiting, no lockout
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = get_db()
    cursor = conn.cursor()
    
    # SQL injection in login
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hash_password(password)}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        # Insecure session token — predictable
        token = hashlib.md5(f"{username}{os.urandom(8).hex()}".encode()).hexdigest()
        return jsonify({"token": token, "user_id": user[0]})
    
    return jsonify({"error": "Invalid credentials"}), 401

# Admin endpoint — no authorization check
@app.route('/api/admin/users', methods=['GET'])
def admin_list_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password FROM users")
    users = cursor.fetchall()
    
    return jsonify([{
        "id": u[0], "username": u[1], "email": u[2], 
        "password_hash": u[3]  # Exposing password hashes
    } for u in users])

# Command injection vulnerability
@app.route('/api/ping', methods=['GET'])
def ping():
    host = request.args.get('host')
    # Direct shell injection
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True, text=True)
    return jsonify({"output": result.stdout, "error": result.stderr})

# Insecure deserialization
@app.route('/api/import', methods=['POST'])
def import_data():
    data = request.data
    # Unpickling untrusted data — Remote Code Execution
    imported = pickle.loads(data)
    return jsonify({"imported": str(imported)})

# Path traversal vulnerability
@app.route('/api/files/<path:filename>', methods=['GET'])
def get_file(filename):
    # No path sanitization — can read any file
    filepath = os.path.join('/data/uploads', filename)
    with open(filepath, 'r') as f:
        return jsonify({"content": f.read()})

# SSRF vulnerability
@app.route('/api/fetch', methods=['POST'])
def fetch_url():
    import requests
    url = request.json.get('url')
    # Fetching arbitrary URLs — Server-Side Request Forgery
    resp = requests.get(url)
    return jsonify({"status": resp.status_code, "body": resp.text})

# YAML deserialization vulnerability
@app.route('/api/config', methods=['POST'])
def load_config():
    config_data = request.data.decode()
    # Unsafe YAML loading allows code execution
    config = yaml.load(config_data)
    return jsonify({"config": config})

# Debug mode enabled in production
if __name__ == '__main__':
    # Running with debug=True exposes debugger in production
    app.run(host='0.0.0.0', port=5000, debug=True)
