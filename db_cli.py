#!/usr/bin/env python3
"""
Enterprise Database CLI Utility

Provides command-line interface for database management including:
- Health checks and monitoring
- Migration management
- Database information and diagnostics
- Connection testing
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import click
from tabulate import tabulate
from sqlalchemy import text

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from qtinfra.infra.db import DatabaseManager, DatabaseConfig
    from qtinfra.infra.config import settings
except ImportError as e:
    click.echo(f"Error importing database modules: {e}", err=True)
    sys.exit(1)


@click.group()
@click.pass_context
def cli(ctx):
    """Enterprise Database Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['start_time'] = time.time()


@cli.command()
@click.option('--engine', default='primary', help='Database engine to check')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def health(ctx, engine, output_format):
    """Perform database health check"""
    
    async def check_health():
        try:
            manager = DatabaseManager()
            await manager.initialize()
            
            health_status = await manager.health_check(engine)
            
            if output_format == 'json':
                click.echo(json.dumps(health_status, indent=2))
            else:
                # Format as table
                status_color = {
                    'healthy': 'green',
                    'degraded': 'yellow',
                    'slow': 'yellow',
                    'unavailable': 'red'
                }.get(health_status['status'], 'white')
                
                click.echo(f"Database Health Check - Engine: {engine}")
                click.echo("=" * 50)
                click.echo(f"Status: {click.style(health_status['status'].upper(), fg=status_color)}")
                click.echo(f"Response Time: {health_status['response_time']:.3f}s")
                click.echo(f"Timestamp: {health_status['timestamp']}")
                
                if health_status['pool_status']:
                    click.echo("\nConnection Pool Status:")
                    pool_data = [
                        ['Metric', 'Value'],
                        ['Pool Size', health_status['pool_status'].get('size', 'N/A')],
                        ['Checked In', health_status['pool_status'].get('checked_in', 'N/A')],
                        ['Checked Out', health_status['pool_status'].get('checked_out', 'N/A')],
                        ['Overflow', health_status['pool_status'].get('overflow', 'N/A')],
                        ['Invalid', health_status['pool_status'].get('invalid', 'N/A')]
                    ]
                    click.echo(tabulate(pool_data, headers='firstrow', tablefmt='grid'))
                
                if health_status['errors']:
                    click.echo(f"\nErrors:")
                    for error in health_status['errors']:
                        click.echo(f"  • {click.style(error, fg='red')}")
            
            await manager.shutdown()
            
        except Exception as e:
            click.echo(f"Health check failed: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(check_health())


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def info(ctx, output_format):
    """Get comprehensive database information"""
    
    async def get_info():
        try:
            manager = DatabaseManager()
            await manager.initialize()
            
            db_info = await manager.get_database_info()
            
            if output_format == 'json':
                click.echo(json.dumps(db_info, indent=2, default=str))
            else:
                click.echo("Database Information")
                click.echo("=" * 50)
                
                # General info
                click.echo(f"Uptime: {db_info['uptime']:.2f} seconds")
                
                # Configuration
                click.echo("\nConfiguration:")
                config_data = [
                    ['Setting', 'Value'],
                    ['Pool Size', db_info['config']['pool_size']],
                    ['Max Overflow', db_info['config']['max_overflow']],
                    ['Pool Timeout', db_info['config']['pool_timeout']],
                    ['Query Timeout', db_info['config']['query_timeout']],
                    ['SSL Enabled', db_info['config']['ssl_enabled']],
                    ['Connection Strategy', db_info['config']['connection_strategy']]
                ]
                click.echo(tabulate(config_data, headers='firstrow', tablefmt='grid'))
                
                # Circuit breaker status
                cb_status = db_info['circuit_breaker']
                click.echo(f"\nCircuit Breaker:")
                click.echo(f"  State: {cb_status['state']}")
                click.echo(f"  Failure Count: {cb_status['failure_count']}")
                click.echo(f"  Last Failure: {cb_status['last_failure_time']}")
                
                # Engine status
                click.echo(f"\nEngines ({len(db_info['engines'])}):")
                for engine_name, engine_info in db_info['engines'].items():
                    status_color = {
                        'healthy': 'green',
                        'degraded': 'yellow', 
                        'slow': 'yellow',
                        'unavailable': 'red'
                    }.get(engine_info['status'], 'white')
                    
                    click.echo(f"  {engine_name}: {click.style(engine_info['status'], fg=status_color)} "
                             f"({engine_info['response_time']:.3f}s)")
            
            await manager.shutdown()
            
        except Exception as e:
            click.echo(f"Failed to get database info: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(get_info())


@cli.command()
@click.option('--interval', default=5, help='Check interval in seconds')
@click.option('--max-checks', default=0, help='Maximum number of checks (0 for infinite)')
@click.pass_context
def monitor(ctx, interval, max_checks):
    """Monitor database health continuously"""
    
    async def monitor_health():
        try:
            manager = DatabaseManager()
            await manager.initialize()
            
            check_count = 0
            
            click.echo(f"Starting database monitoring (interval: {interval}s)")
            click.echo("Press Ctrl+C to stop")
            click.echo("=" * 60)
            
            while max_checks == 0 or check_count < max_checks:
                try:
                    health_status = await manager.health_check()
                    
                    status_color = {
                        'healthy': 'green',
                        'degraded': 'yellow',
                        'slow': 'yellow', 
                        'unavailable': 'red'
                    }.get(health_status['status'], 'white')
                    
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    response_time = health_status['response_time']
                    
                    click.echo(f"[{timestamp}] Status: {click.style(health_status['status'].upper(), fg=status_color)} "
                             f"Response: {response_time:.3f}s")
                    
                    if health_status['errors']:
                        for error in health_status['errors']:
                            click.echo(f"  ⚠ {click.style(error, fg='red')}")
                    
                    check_count += 1
                    
                    if max_checks == 0 or check_count < max_checks:
                        await asyncio.sleep(interval)
                    
                except KeyboardInterrupt:
                    click.echo("\nMonitoring stopped by user")
                    break
                except Exception as e:
                    click.echo(f"Monitor error: {e}", err=True)
                    await asyncio.sleep(interval)
            
            await manager.shutdown()
            
        except Exception as e:
            click.echo(f"Monitoring failed: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(monitor_health())


@cli.command()
@click.option('--timeout', default=10, help='Connection timeout in seconds')
@click.pass_context
def test_connection(ctx, timeout):
    """Test database connection"""
    
    async def test_conn():
        try:
            click.echo("Testing database connection...")
            
            manager = DatabaseManager()
            start_time = time.time()
            
            await asyncio.wait_for(manager.initialize(), timeout=timeout)
            
            # Test basic query
            async with manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                test_value = result.scalar()
            
            connection_time = time.time() - start_time
            
            if test_value == 1:
                click.echo(f"{click.style('✓', fg='green')} Connection successful!")
                click.echo(f"  Connection time: {connection_time:.3f}s")
                click.echo(f"  Database URL: {manager.config.url}")
                click.echo(f"  SSL Enabled: {manager.config.enable_ssl}")
            else:
                click.echo(f"{click.style('✗', fg='red')} Connection test failed - unexpected result")
                sys.exit(1)
            
            await manager.shutdown()
            
        except asyncio.TimeoutError:
            click.echo(f"{click.style('✗', fg='red')} Connection timeout after {timeout}s")
            sys.exit(1)
        except Exception as e:
            click.echo(f"{click.style('✗', fg='red')} Connection failed: {e}")
            sys.exit(1)
    
    asyncio.run(test_conn())


@cli.command()
@click.option('--message', '-m', help='Migration message')
@click.option('--autogenerate/--no-autogenerate', default=True, help='Auto-generate migration')
def create_migration(message, autogenerate):
    """Create a new database migration"""
    try:
        import alembic.config
        import alembic.command
        
        if not message:
            message = click.prompt('Migration message')
        
        alembic_cfg = alembic.config.Config('alembic.ini')
        
        if autogenerate:
            alembic.command.revision(alembic_cfg, message=message, autogenerate=True)
        else:
            alembic.command.revision(alembic_cfg, message=message)
        
        click.echo(f"{click.style('✓', fg='green')} Migration created: {message}")
        
    except ImportError:
        click.echo("Alembic not installed. Install with: pip install alembic", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to create migration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--revision', default='head', help='Target revision (default: head)')
def migrate(revision):
    """Run database migrations"""
    try:
        import alembic.config
        import alembic.command
        
        alembic_cfg = alembic.config.Config('alembic.ini')
        alembic.command.upgrade(alembic_cfg, revision)
        
        click.echo(f"{click.style('✓', fg='green')} Migrations applied to {revision}")
        
    except ImportError:
        click.echo("Alembic not installed. Install with: pip install alembic", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--revision', required=True, help='Target revision to downgrade to')
def downgrade(revision):
    """Downgrade database to a specific revision"""
    try:
        import alembic.config
        import alembic.command
        
        if not click.confirm(f'Are you sure you want to downgrade to {revision}?'):
            click.echo("Downgrade cancelled")
            return
        
        alembic_cfg = alembic.config.Config('alembic.ini')
        alembic.command.downgrade(alembic_cfg, revision)
        
        click.echo(f"{click.style('✓', fg='green')} Database downgraded to {revision}")
        
    except ImportError:
        click.echo("Alembic not installed. Install with: pip install alembic", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Downgrade failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def migration_history():
    """Show migration history"""
    try:
        import alembic.config
        import alembic.command
        
        alembic_cfg = alembic.config.Config('alembic.ini')
        alembic.command.history(alembic_cfg, verbose=True)
        
    except ImportError:
        click.echo("Alembic not installed. Install with: pip install alembic", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to show history: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()