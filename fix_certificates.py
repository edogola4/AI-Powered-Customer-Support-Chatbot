import os
import ssl

# Get the path to the certifi package
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
print(f"Certificate path set to: {certifi.where()}")

# Test SSL connection
try:
    ssl.get_server_certificate(('www.python.org', 443))
    print("SSL verification successful!")
except Exception as e:
    print(f"SSL verification failed: {e}")