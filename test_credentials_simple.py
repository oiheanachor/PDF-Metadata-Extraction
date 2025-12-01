#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Azure AI Foundry Credential Checker
No Azure CLI required - just tests API key directly
"""

import os
import sys

print("=" * 80)
print("AZURE AI FOUNDRY - SIMPLE CREDENTIAL CHECK")
print("=" * 80)

# Step 1: Check environment variables
print("\n1. Checking environment variables...")

endpoint = os.getenv("AZURE_AI_ENDPOINT")
api_key = os.getenv("AZURE_AI_KEY")

if not endpoint:
    print("❌ AZURE_AI_ENDPOINT not set\n")
    print("Please run this in PowerShell:")
    print('$env:AZURE_AI_ENDPOINT = "YOUR_ENDPOINT_HERE"')
    print("\nTo find your endpoint:")
    print("1. Go to https://ai.azure.com")
    print("2. Click on your project: aiproject003")
    print("3. Look for 'Project endpoint' or 'Endpoint' on the main page")
    print("4. Copy the EXACT URL shown")
    print("\nCommon formats:")
    print("  - https://PROJECT.REGION.inference.ai.azure.com")
    print("  - https://PROJECT.REGION.services.ai.azure.com")
    print("  - https://PROJECT.inference.ml.azure.com")
    sys.exit(1)
else:
    print(f"✓ AZURE_AI_ENDPOINT found: {endpoint}")

if not api_key:
    print("❌ AZURE_AI_KEY not set\n")
    print("Please run this in PowerShell:")
    print('$env:AZURE_AI_KEY = "YOUR_API_KEY_HERE"')
    print("\nTo find your API key:")
    print("1. Go to https://ai.azure.com")
    print("2. Click on your project: aiproject003")
    print("3. Find 'Keys and Endpoints' or 'API Keys' in the menu")
    print("4. Copy your key")
    sys.exit(1)
else:
    # Show partial key for verification
    if len(api_key) > 20:
        masked = f"{api_key[:8]}...{api_key[-4:]}"
    else:
        masked = f"{api_key[:4]}...{api_key[-2:]}"
    print(f"✓ AZURE_AI_KEY found: {masked}")

# Step 2: Try importing SDK
print("\n2. Checking azure-ai-inference installation...")
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem
    from azure.core.credentials import AzureKeyCredential
    print("✓ azure-ai-inference is installed")
except ImportError as e:
    print(f"❌ azure-ai-inference not installed\n")
    print("Install with:")
    print("pip install azure-ai-inference")
    sys.exit(1)

# Step 3: Create client
print("\n3. Creating client...")
try:
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )
    print("✓ Client created")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    sys.exit(1)

# Step 4: Test with simple call
print("\n4. Testing API call...")
print("   (This will make a real API request)")

try:
    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Reply with just the word 'SUCCESS' and nothing else.")
        ],
        model="claude-sonnet-4-20250514",
        max_tokens=20,
        temperature=0
    )
    
    result = response.choices[0].message.content
    print(f"\n✅ API CALL SUCCESSFUL!")
    print(f"   Response: {result}")
    print(f"\n🎉 Your credentials are working correctly!")
    print(f"   Endpoint: {endpoint}")
    print(f"   Model: claude-sonnet-4-20250514")
    print(f"\nYou can now run:")
    print(f'   python rev_extractor_pymupdf_claude.py "YOUR_FOLDER"')

except Exception as e:
    error_str = str(e)
    print(f"\n❌ API CALL FAILED")
    print(f"   Error: {error_str}\n")
    
    # Diagnose the error
    if "401" in error_str or "Unauthorized" in error_str:
        print("⚠️  AUTHENTICATION ERROR (401)")
        print("\nMost likely causes:")
        print("1. Wrong API key")
        print("2. Wrong endpoint URL")
        print("3. Key expired")
        print("\n📋 Try these fixes:")
        print("\nFix 1: Get NEW API key")
        print("  1. Go to https://ai.azure.com → aiproject003")
        print("  2. Find 'Keys and Endpoints'")
        print("  3. Click 'Regenerate primary key'")
        print("  4. Copy the new key")
        print("  5. Set it: $env:AZURE_AI_KEY = 'new_key'")
        print("\nFix 2: Try different endpoint formats")
        print(f"  Current: {endpoint}")
        print("\n  Try each of these:")
        print("  $env:AZURE_AI_ENDPOINT = 'https://aifoundrydevtest003.services.ai.azure.com'")
        print("  $env:AZURE_AI_ENDPOINT = 'https://aifoundrydevtest003.uksouth.services.ai.azure.com'")
        print("  $env:AZURE_AI_ENDPOINT = 'https://aifoundrydevtest003.uksouth.inference.ai.azure.com'")
        print("\n  Test each with: python test_credentials_simple.py")
        
    elif "404" in error_str or "Not Found" in error_str:
        print("⚠️  MODEL NOT FOUND (404)")
        print("\nThe model 'claude-sonnet-4-20250514' might not be deployed.")
        print("\n📋 Check your deployed models:")
        print("  1. Go to https://ai.azure.com → aiproject003")
        print("  2. Find 'Deployments' or 'Models'")
        print("  3. Look for Claude models")
        print("  4. Note the exact model name")
        print("\n  Common names:")
        print("  - claude-sonnet-4-20250514")
        print("  - claude-sonnet-4")
        print("  - claude-3-5-sonnet")
        
    elif "403" in error_str or "Forbidden" in error_str:
        print("⚠️  ACCESS FORBIDDEN (403)")
        print("\nYour key might not have permission to use this model.")
        print("\n📋 Check permissions:")
        print("  1. Go to https://ai.azure.com → aiproject003")
        print("  2. Check your role/permissions")
        print("  3. Verify the model is deployed and accessible")
        
    else:
        print("⚠️  UNKNOWN ERROR")
        print("\n📋 General troubleshooting:")
        print("  1. Verify you're logged into the correct Azure account")
        print("  2. Check the project is active (not suspended)")
        print("  3. Verify you have quota/credits")
        print("  4. Try the Azure portal: https://portal.azure.com")
    
    sys.exit(1)

print("\n" + "=" * 80)
