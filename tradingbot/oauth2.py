import requests
import json
import os
from datetime import datetime, timedelta

TOKEN_FILE = "questrade_tokens.json"

class QuestradeAuth:
    def __init__(self):
        self.load_tokens()
    
    def load_tokens(self):
        """Load tokens from file or initialize"""
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE) as f:
                tokens = json.load(f)
                self.refresh_token = tokens["refresh_token"]
                self.access_token = tokens["access_token"]
                self.api_server = tokens["api_server"]
                self.token_expiry = datetime.fromisoformat(tokens["token_expiry"])
        else:
            self.refresh_token = "kF-0jttHOdpo_MFj6oYJCXw-dv4HYbMT0"
            self.refresh_access_token()
    
    def save_tokens(self):
        """Save current tokens to file"""
        tokens = {
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "api_server": self.api_server,
            "token_expiry": self.token_expiry.isoformat()
        }
        with open(TOKEN_FILE, 'w') as f:
            json.dump(tokens, f)
    
    def refresh_access_token(self):
        """Refresh tokens and save them"""
        try:
            url = f"https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token={self.refresh_token}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.api_server = data["api_server"]
            self.token_expiry = datetime.now() + timedelta(seconds=data["expires_in"] - 60)  # 1 minute buffer
            
            self.save_tokens()
            print(f"Token refreshed | Valid until: {self.token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        except Exception as e:
            print(f"Failed to refresh token: {e}")
            return False
    
    def make_request(self, endpoint):
        """Auto-refreshing API call"""
        # Check token expiry BEFORE making request
        if datetime.now() >= self.token_expiry - timedelta(minutes=1):  # Refresh 1 minute before expiry
            if not self.refresh_access_token():
                raise Exception("Failed to refresh access token")
        
        url = f"{self.api_server}{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:  # If still unauthorized after refresh
                if self.refresh_access_token():  # Try one more refresh
                    response = requests.get(url, headers={"Authorization": f"Bearer {self.access_token}"})
                    response.raise_for_status()
                    return response.json()
            raise  # Re-raise other errors

# Usage
if __name__ == "__main__":
    qt = QuestradeAuth()  # Automatically loads/saves tokens
    # Example API call
    try:
        accounts = qt.make_request("/v1/accounts")
        print("Accounts:", accounts)
    except Exception as e:
        print(f"API Error: {e}")


def main():
    qt = QuestradeAuth()
    accounts = qt.make_request("/v1/accounts")
    print(accounts)