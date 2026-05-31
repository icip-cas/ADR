import json
import requests


def call_sandbox_api(code, test_cases, url, timeout=30):
    # Config
    compile_timeout = timeout
    run_timeout = timeout
    API_TIMEOUT = 10
    
    # Output format
    result_status = -1
    metadata = {
        'response_data': None,
        'response_status_code': None,
        'pass_rate': None,
    }

    try:
        payload = json.dumps(
            {
                "completion": f'```python\n{code}\n```',
                'config': {
                    'language': 'python', 
                    'compile_timeout': compile_timeout, 
                    'run_timeout': run_timeout, 
                    'provided_data': {
                        'test_cases': test_cases
                    }, 
                    'extra': {
                        'run_all_cases': True, 
                        'total_timeout': 30
                    },
                }
                                
            }
        )
            
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        request_timeout = compile_timeout + run_timeout + API_TIMEOUT

        response = requests.post(
            url,
            headers=headers,
            data=payload,
            timeout=request_timeout,  # Use the calculated timeout
        )
        
        response_data = response.json()
        
        metadata['response_data'] = str(response_data)
        
        if len(metadata['response_data']) > 10000:
            metadata['response_data'] = metadata['response_data'][:5000] + '... [truncated]' + metadata['response_data'][-5000:]
        
        metadata['response_status_code'] = response.status_code
        if response.status_code == 200:
            accepted = response_data.get("accepted", False)
            if accepted:
                result_status = True
            else:
                result_status = False
            
            # Calculate pass rate
            cases = response_data.get('tests', [])
            total_cases = len(cases)
            passed_cases = sum(1 for test in cases if test and test.get('passed', False))
            pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
            metadata['pass_rate'] = pass_rate
        else:
            result_status = -2
    except Exception as e:
        print(f"Error calling sandbox API: {e}")

    return result_status, metadata


def call_sandbox_api_pytest(code, test_code, url, timeout=60):
    # Output format
    result_status = -1
    metadata = {
        'response_data': None,
        'response_status_code': None,
        'pass_rate': None,
    }

    try:
        payload = {
            'code': code + '\n\n' + test_code, 
            'language': 'pytest',
            "compile_timeout": timeout,
            "run_timeout": timeout,
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response_data = response.json()
        
        metadata['response_data'] = str(response_data)
        if len(metadata['response_data']) > 10000:
            metadata['response_data'] = metadata['response_data'][:5000] + '... [truncated]' + metadata['response_data'][-5000:]
        
        metadata['response_status_code'] = response.status_code
        
        if response.status_code == 200:
            if response_data.get("status", "") == "Success":
                result_status = True
            else:
                result_status = False
        else:
            result_status = -2
    except requests.exceptions.RequestException as e:
        print(f"Error calling sandbox API: {e}")
    
    return result_status, metadata


def call_sandbox_api_python(code, url):
    # Output format
    result_status = -1
    metadata = {
        'response_data': None,
        'response_status_code': None,
        'pass_rate': None,
    }
    
    try:
        payload = {
            'code': code, 
            'language': 'python',
            "compile_timeout": 60,
            "run_timeout": 60,
        }
    
        response = requests.post(url, json=payload, timeout=120)
        response_data = response.json()
        
        metadata['response_data'] = str(response_data)
        if len(metadata['response_data']) > 10000:
            metadata['response_data'] = metadata['response_data'][:5000] + '... [truncated]' + metadata['response_data'][-5000:]
        
        metadata['response_status_code'] = response.status_code
        
        if response.status_code == 200:
            if response_data.get("status", "") == "Success":
                result_status = True
            else:
                result_status = False
        else:
            result_status = -2
    except requests.exceptions.RequestException as e:
        print(f"Error calling sandbox API: {e}")
    
    return result_status, metadata