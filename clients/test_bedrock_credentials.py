#!/usr/bin/env python3
"""
AWS Bedrock Claude Credential Tester
This script verifies your AWS credentials and Claude model access.
"""

import os
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from anthropic import AnthropicBedrock


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_environment_variables():
    """Check if required environment variables are set"""
    print_section("ENVIRONMENT VARIABLES CHECK")
    
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY", 
        "AWS_REGION",
        "CLAUDE_BEDROCK_MODEL_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "SECRET" in var or "KEY" in var:
                masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print(f"\n✅ All environment variables are set!")
        return True


def test_basic_aws_access():
    """Test basic AWS credentials and access"""
    print_section("BASIC AWS ACCESS TEST")
    
    try:
        # Test with STS (Security Token Service)
        sts = boto3.client('sts', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        identity = sts.get_caller_identity()
        
        print(f"✅ AWS Credentials Valid!")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}")
        print(f"   User ID: {identity['UserId']}")
        return True
        
    except NoCredentialsError:
        print("❌ No AWS credentials found!")
        print("   Make sure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set.")
        return False
        
    except PartialCredentialsError:
        print("❌ Incomplete AWS credentials!")
        print("   Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.")
        return False
        
    except ClientError as e:
        print(f"❌ AWS Access Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_bedrock_access():
    """Test AWS Bedrock service access"""
    print_section("BEDROCK SERVICE ACCESS TEST")
    
    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        bedrock = boto3.client('bedrock', region_name=region)
        
        # Test basic Bedrock access
        models = bedrock.list_foundation_models()
        print(f"✅ Bedrock service accessible in region: {region}")
        print(f"   Found {len(models['modelSummaries'])} total models")
        
        # Check for Claude models specifically
        claude_models = [m for m in models['modelSummaries'] if 'claude' in m['modelId'].lower()]
        
        if claude_models:
            print(f"✅ Found {len(claude_models)} Claude models:")
            for model in claude_models:
                print(f"   - {model['modelId']}")
        else:
            print("❌ No Claude models found!")
            print("   This might mean Claude isn't available in your region.")
        
        return True, claude_models
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UnauthorizedOperation':
            print("❌ Access Denied to Bedrock!")
            print("   Your AWS user/role doesn't have Bedrock permissions.")
        elif error_code == 'OptInRequired':
            print("❌ Bedrock access not enabled!")
            print("   You need to enable Bedrock in the AWS console first.")
        else:
            print(f"❌ Bedrock Error ({error_code}): {e}")
        return False, []
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, []


def test_model_access():
    """Test access to specific Claude model"""
    print_section("CLAUDE MODEL ACCESS TEST")
    
    model_id = os.getenv('CLAUDE_BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        bedrock = boto3.client('bedrock', region_name=region)
        
        # Get model details
        model_info = bedrock.get_foundation_model(modelIdentifier=model_id)
        print(f"✅ Model {model_id} is accessible!")
        print(f"   Model Name: {model_info['modelDetails']['modelName']}")
        print(f"   Provider: {model_info['modelDetails']['providerName']}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceNotFoundException':
            print(f"❌ Model {model_id} not found!")
            print("   Check your CLAUDE_BEDROCK_MODEL_ID environment variable.")
        elif error_code == 'AccessDeniedException':
            print(f"❌ Access denied to model {model_id}!")
            print("   You may need to request access to this model in the Bedrock console.")
        else:
            print(f"❌ Model access error ({error_code}): {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_bedrock_runtime():
    """Test Bedrock Runtime (actual model invocation)"""
    print_section("BEDROCK RUNTIME TEST")
    
    model_id = os.getenv('CLAUDE_BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    try:
        # Test with boto3 directly
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        
        # Simple test message
        test_prompt = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with exactly: TEST SUCCESSFUL"
                }
            ]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(test_prompt)
        )
        
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']
        
        print(f"✅ Bedrock Runtime working!")
        print(f"   Model Response: {content}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print(f"❌ Access denied to invoke model {model_id}!")
            print("   You need to request access to this model in the Bedrock console.")
            print("   Go to: AWS Bedrock Console > Model access > Request model access")
        elif error_code == 'ValidationException':
            print(f"❌ Invalid request format for model {model_id}!")
            print("   The model might use a different request format.")
        else:
            print(f"❌ Runtime error ({error_code}): {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_anthropic_bedrock_library():
    """Test the AnthropicBedrock library"""
    print_section("ANTHROPIC BEDROCK LIBRARY TEST")
    
    try:
        client = AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1")
        )
        
        model_id = os.getenv('CLAUDE_BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        
        response = client.messages.create(
            model=model_id,
            max_tokens=50,
            messages=[
                {
                    "role": "user", 
                    "content": "Say exactly: ANTHROPIC LIBRARY TEST SUCCESSFUL"
                }
            ]
        )
        
        content = response.content[0].text
        print(f"✅ AnthropicBedrock library working!")
        print(f"   Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ AnthropicBedrock library error: {e}")
        return False


def suggest_solutions():
    """Suggest common solutions for issues"""
    print_section("TROUBLESHOOTING SUGGESTIONS")
    
    print("If you're having issues, try these steps:")
    print()
    print("1. 🔐 CREDENTIALS:")
    print("   - Make sure your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are correct")
    print("   - These should be for an IAM user with Bedrock permissions")
    print()
    print("2. 🌍 REGION:")
    print("   - Try AWS_REGION='us-west-2' (Oregon) - most reliable for Claude")
    print("   - Or try AWS_REGION='us-east-1' (Virginia)")
    print()
    print("3. 🔑 MODEL ACCESS:")
    print("   - Go to AWS Bedrock Console > Model access")
    print("   - Request access to 'Anthropic Claude 3 Sonnet'")
    print("   - Wait for approval (usually takes a few minutes)")
    print()
    print("4. 📋 PERMISSIONS:")
    print("   - Your IAM user needs these policies:")
    print("     - AmazonBedrockFullAccess (or custom Bedrock permissions)")
    print()
    print("5. 🔄 ALTERNATIVE:")
    print("   - Consider using Claude's direct API instead of Bedrock")
    print("   - Set ANTHROPIC_API_KEY instead of AWS credentials")


def main():
    """Run all credential tests"""
    print("🧪 AWS Bedrock Claude Credential Tester")
    print("This script will test your AWS Bedrock setup for Claude access.")
    
    results = {
        'env_vars': check_environment_variables(),
        'aws_access': False,
        'bedrock_access': False, 
        'model_access': False,
        'runtime_test': False,
        'library_test': False
    }
    
    if results['env_vars']:
        results['aws_access'] = test_basic_aws_access()
        
        if results['aws_access']:
            bedrock_ok, claude_models = test_bedrock_access()
            results['bedrock_access'] = bedrock_ok
            
            if results['bedrock_access']:
                results['model_access'] = test_model_access()
                
                if results['model_access']:
                    results['runtime_test'] = test_bedrock_runtime()
                    results['library_test'] = test_anthropic_bedrock_library()
    
    # Summary
    print_section("TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your Claude Bedrock setup should work.")
    else:
        print("⚠️  Some tests failed. See suggestions below.")
        suggest_solutions()


if __name__ == "__main__":
    main()
