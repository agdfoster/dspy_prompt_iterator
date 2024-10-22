import pandas as pd
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
import os
import json
from google.auth.exceptions import MalformedError
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Set up credentials
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if not SERVICE_ACCOUNT_FILE:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")

try:
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")

    with open(SERVICE_ACCOUNT_FILE, 'r') as f:
        service_account_info = json.load(f)

    required_fields = ['token_uri', 'client_email', 'private_key']
    missing_fields = [field for field in required_fields if field not in service_account_info]
    
    if missing_fields:
        raise MalformedError(f"Service account info is missing required fields: {', '.join(missing_fields)}")

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the service account key file exists at the specified path.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: The service account file is not a valid JSON file.")
    print("Please check the file contents and ensure it's a properly formatted JSON.")
    exit(1)
except MalformedError as e:
    print(f"Error: {e}")
    print("Please generate a new service account key from the Google Cloud Console.")
    exit(1)

# Build the Google Sheets API service
service = build('sheets', 'v4', credentials=creds)
drive_service = build('drive', 'v3', credentials=creds)

def create_google_sheet(title):
    """Create a new Google Sheet and return its ID."""
    sheet_metadata = service.spreadsheets().create(body={
        'properties': {'title': title}
    }).execute()
    print(f"{Fore.GREEN}Sheet created with ID: {sheet_metadata['spreadsheetId']}{Style.RESET_ALL}")
    return sheet_metadata['spreadsheetId']

def df_to_sheet(df, sheet_id, sheet_name='Sheet1'):
    """Write a DataFrame to a specific sheet in a Google Sheet."""
    values = [df.columns.tolist()] + df.values.tolist()
    body = {'values': values}
    result = service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f'{sheet_name}!A1',
        valueInputOption='RAW',
        body=body
    ).execute()
    print(f"{Fore.CYAN}{result.get('updatedCells')} cells updated.{Style.RESET_ALL}")

def share_sheet(sheet_id, email):
    """Share the Google Sheet with a specific email address."""
    try:
        permission = drive_service.permissions().create(
            fileId=sheet_id,
            body={'type': 'user', 'role': 'writer', 'emailAddress': email},
            fields='id',
            sendNotificationEmail=False
        ).execute()
        print(f"{Fore.GREEN}Sheet shared successfully with {email}. Permission ID: {permission.get('id')}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error sharing sheet: {str(e)}{Style.RESET_ALL}")
        if hasattr(e, 'content'):
            print(f"{Fore.RED}Error details: {e.content.decode('utf-8')}{Style.RESET_ALL}")

def save_df_to_google_sheet(df, sheet_name, email_to_share=None):
    """Save a DataFrame to a new Google Sheet and optionally share it."""
    sheet_id = create_google_sheet(sheet_name)
    df_to_sheet(df, sheet_id)
    
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    print(f"{Fore.YELLOW}Sheet URL: {sheet_url}{Style.RESET_ALL}")
    
    if email_to_share:
        share_sheet(sheet_id, email_to_share)
        verify_sharing(sheet_id, email_to_share)
    
    return sheet_url

def verify_sharing(sheet_id, email):
    """Verify if the sheet is shared with the specified email."""
    try:
        permissions = drive_service.permissions().list(fileId=sheet_id).execute()
        for permission in permissions.get('permissions', []):
            if permission.get('emailAddress') == email:
                print(f"{Fore.GREEN}Sharing verified: Sheet is shared with {email}{Style.RESET_ALL}")
                return
        print(f"{Fore.YELLOW}Warning: Sheet does not appear to be shared with {email}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error verifying sharing: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Test the functions
    test_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    sheet_name = "Test Sheet"
    email_to_share = "alex.foster@invisible.email"  # Update this to your actual email
    
    print(f"{Fore.MAGENTA}Starting test...{Style.RESET_ALL}")
    sheet_url = save_df_to_google_sheet(test_df, sheet_name, email_to_share)
    print(f"{Fore.MAGENTA}Test complete. Final Sheet URL: {sheet_url}{Style.RESET_ALL}")
