import uvicorn
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Run DeepSNP3D API server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload (development only)')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()