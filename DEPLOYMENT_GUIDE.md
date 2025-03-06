# Deployment Guide

## Prerequisites
- Vercel account
- Firebase project with service account credentials

## Steps

1. **Convert Firebase Credentials to Base64**
   - Open terminal and navigate to your project directory
   - Run: `base64 -i profitpath-e85f2-firebase-adminsdk-fbsvc-32920b09c0.json`
   - Copy the contents of firebase_creds_base64.txt

2. **Set Environment Variables in Vercel**
   - Go to your Vercel project settings
   - Navigate to Environment Variables section
   - Add new variable:
     - Key: FIREBASE_CREDENTIALS_BASE64
     - Value: [paste base64 content]

3. **Deploy Application**
   - Install Vercel CLI if not already installed: `npm i -g vercel`
   - Run deployment command: `vercel --prod`

4. **Verify Deployment**
   - Check Vercel dashboard for deployment status
   - Test your application endpoints

## Security Considerations
- Never commit the Firebase credentials JSON file to version control
- Use environment variables for sensitive information
- Regularly rotate your Firebase service account keys

## Troubleshooting
- If Firebase initialization fails, verify:
  - Correct base64 encoding of credentials
  - Environment variable is properly set in Vercel
  - Firebase project permissions are correct