#!/usr/bin/env python3

import os
import sys
import time
import atexit
import signal
import logging
import threading
import questionary
from datetime import datetime
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt, Confirm
from typing import Dict, List, Optional

from temp_mail_manager import (
    TempMailManager,
    GuerrillaMailAPI,
    DisposableMailAPI,
    YopMailAPI,
    TempMailOrgAPI,
    TempMailAPI
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger('temp_mail_cli')

# Initialize console
console = Console()

# Initialize manager
active_manager = TempMailManager()

# Global state
is_running = True
monitoring_threads = []

def display_goodbye():
    """Display goodbye message"""
    try:
        console.print(Panel(
            "[bold blue]Thank you for using TempMail Manager![/]\n"
            "[italic]Have a great day![/]",
            title="Goodbye!",
            border_style="green"
        ))
    except Exception:
        pass  # Ignore errors during cleanup

def cleanup(signum=None, frame=None):
    """Cleanup function to handle program exit"""
    try:
        # Save email data
        if active_manager:
            active_manager.save_data()
            
        # Stop all monitoring threads
        for thread, stop_event in monitoring_threads:
            stop_event.set()
            if thread.is_alive():
                thread.join(timeout=1)
                
        # Clear screen and display goodbye only if not already displayed
        if not hasattr(cleanup, 'goodbye_displayed'):
            display_goodbye()
            cleanup.goodbye_displayed = True
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        
    finally:
        # Ensure we exit
        if signum is not None:
            sys.exit(0)

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    cleanup(signum, frame)

# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup)

def list_emails(active_only=False):
    """
    List emails with optional filtering
    
    Args:
        active_only (bool): If True, return only active emails
    
    Returns:
        List[Dict]: List of email dictionaries
    """
    global active_manager
    
    if active_only:
        # Filter for active emails (you might want to define what makes an email 'active')
        return [
            email for email in active_manager.emails_data 
            if email.get('is_active', True)  # Assuming there's an 'is_active' flag
        ]
    else:
        # Return all emails
        return active_manager.emails_data

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_welcome():
    """Display welcome message"""
    console.print(Panel(
        "[bold blue]Welcome to TempMail Manager![/]\n"
        "[italic]Your One-Stop Email Solution[/]",
        border_style="blue"
    ))

def display_menu():
    """Display main menu"""
    return questionary.select(
        "Choose an option:",
        choices=[
            "1. Generate new email",
            "2. Monitor emails",
            "3. List active emails",
            "4. Check messages",
            "5. Forward email",
            "6. Export emails",
            "7. Delete email",
            "8. Clear screen",
            "9. Exit"
        ],
        use_indicator=True
    ).ask()

def generate_email():
    """Generate a new email address"""
    try:
        # Let user select provider
        provider_choice = questionary.select(
            "Select email provider:",
            choices=[
                "Guerrilla Mail",
                "1secmail",
                "YopMail",
                "Temp-Mail.org"
            ]
        ).ask()
        
        if not provider_choice:
            return
            
        # Create provider instance
        provider = None
        if provider_choice == "Guerrilla Mail":
            provider = GuerrillaMailAPI()
        elif provider_choice == "1secmail":
            provider = DisposableMailAPI()
        elif provider_choice == "YopMail":
            provider = YopMailAPI()
        else:
            provider = TempMailOrgAPI()
            
        # Generate email
        email = provider.generate_email()
        if email:
            # Store provider instance in manager
            if active_manager.add_email(email, provider):
                console.print(f"\nGenerated new email: {email}")
                console.print(f"Provider: {provider.__class__.__name__}")
                
                # Verify provider is stored correctly
                stored_provider = active_manager.get_provider_for_email(email)
                if not stored_provider:
                    console.print("[red]Warning: Provider not stored correctly[/red]")
        else:
            console.print("[red]Failed to generate email[/red]")
            
    except Exception as e:
        logger.error(f"Error generating email: {str(e)}")
        console.print("[red]Failed to generate email. Please try another provider.[/red]")

def select_email_to_monitor():
    """Select an email to monitor from active emails"""
    try:
        if not active_manager.emails_data:
            console.print("[yellow]No active emails found. Generate an email first.[/]")
            return None

        table = Table(title="Active Email Addresses")
        table.add_column("Index", justify="center")
        table.add_column("Email Address", justify="left")
        table.add_column("Provider", justify="center")
        table.add_column("Created At", justify="center")

        for idx, email_data in enumerate(active_manager.emails_data, 1):
            table.add_row(
                str(idx),
                email_data['email'],
                email_data.get('provider', 'Unknown'),
                email_data.get('created_at', 'Unknown')
            )

        console.print(table)

        # Get user selection
        while True:
            choice = Prompt.ask(
                "Select email to monitor",
                default=1,
                show_default=True
            )
            
            if 1 <= choice <= len(active_manager.emails_data):
                selected_email = active_manager.emails_data[choice-1]['email']
                console.print(f"\nSelected: {selected_email}")
                return selected_email
            else:
                console.print("[red]Invalid selection. Please try again.[/]")

    except Exception as e:
        logger.error(f"Error selecting email: {str(e)}")
        console.print("[red]Error selecting email. Please try again.[/]")
        return None

def display_message(message: dict):
    """Display a single message in a nice format"""
    panel = Panel(
        f"From: {message.get('from', 'Unknown')}\n"
        f"Subject: {message.get('subject', 'No Subject')}\n"
        f"Time: {datetime.fromtimestamp(int(message.get('date', 0))).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Body: {message.get('body', '...')}",
        title="New Message",
        title_align="left"
    )
    console.print(panel)

def monitor_email_thread(email: str, stop_event: threading.Event):
    """Monitor email for new messages"""
    try:
        provider = active_manager.get_provider_for_email(email)
        if not provider:
            logger.error(f"No provider found for email: {email}")
            return

        seen_message_ids = set()
        while not stop_event.is_set():
            try:
                messages = provider.get_messages(email)
                for message in messages:
                    msg_id = str(message.get('id'))
                    if msg_id not in seen_message_ids:
                        seen_message_ids.add(msg_id)
                        console.print(Panel(f"""
[bold green]New Message![/bold green]
From: {message.get('from', 'Unknown')}
Subject: {message.get('subject', 'No Subject')}
Date: {message.get('date', 'Unknown')}
""", title=f"Email: {email}"))

                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error monitoring email {email}: {str(e)}")
                time.sleep(30)  # Wait longer on error
                
    except Exception as e:
        logger.error(f"Monitor thread error for {email}: {str(e)}")

def start_monitoring(email: str):
    """Start monitoring an email address"""
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_email_thread, args=(email, stop_event))
    thread.daemon = True
    thread.start()
    monitoring_threads.append((thread, stop_event))
    console.print(f"[green]Started monitoring {email}[/green]")

def check_messages(email: str = None):
    """Check messages for an email address"""
    try:
        if not active_manager.emails:
            console.print("[yellow]No active emails[/yellow]")
            return

        if email is None:
            # Let user select email
            email = questionary.select(
                "Select email to check:",
                choices=list(active_manager.emails.keys())
            ).ask()

        if not email:
            return

        provider = active_manager.get_provider_for_email(email)
        if not provider:
            console.print(f"[red]No provider found for {email}[/red]")
            return

        messages = provider.get_messages(email)
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan")
        table.add_column("From", style="green")
        table.add_column("Subject", style="blue")
        table.add_column("Date", style="magenta")

        valid_messages = []
        if messages and isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                    
                msg_id = str(message.get('id', ''))
                msg_from = str(message.get('from', 'Unknown'))
                msg_subject = str(message.get('subject', 'No Subject'))
                msg_date = str(message.get('date', 'Unknown'))
                
                if msg_id:
                    valid_messages.append(message)
                    table.add_row(
                        msg_id,
                        msg_from,
                        msg_subject,
                        msg_date
                    )

        if valid_messages:
            console.print(table)
            
            # Let user select message to view details
            msg_id = questionary.select(
                "Select message to view details:",
                choices=[str(msg.get('id', '')) for msg in valid_messages]
            ).ask()
            
            if msg_id:
                message = provider.get_message(email, msg_id)
                if message:
                    console.print(Panel(f"""
[bold]From:[/bold] {str(message.get('from', 'Unknown'))}
[bold]Subject:[/bold] {str(message.get('subject', 'No Subject'))}
[bold]Date:[/bold] {str(message.get('date', 'Unknown'))}
[bold]Body:[/bold]
{str(message.get('body', 'No Body'))}
""", title="Message Details"))
                else:
                    console.print("[red]Failed to retrieve message details[/red]")
        else:
            console.print("[yellow]No messages found[/yellow]")

    except Exception as e:
        logger.error(f"Error checking messages: {str(e)}")
        console.print("[red]Error checking messages[/red]")

def monitor_emails():
    """Monitor emails for new messages"""
    try:
        if not active_manager.emails:
            console.print("[red]No active emails to monitor[/red]")
            return

        # Let user select emails to monitor
        emails = questionary.checkbox(
            "Select emails to monitor:",
            choices=list(active_manager.emails.keys())
        ).ask()

        if not emails:
            return

        for email in emails:
            start_monitoring(email)

        console.print("[green]Press Ctrl+C to stop monitoring[/green]")
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping email monitoring...[/yellow]")
            
    except Exception as e:
        logger.error(f"Error in monitor_emails: {str(e)}")
        console.print("[red]Error monitoring emails[/red]")

def stop_monitoring(email_address: str):
    """Stop monitoring an email address"""
    for thread, event in monitoring_threads[:]:
        event.set()
        thread.join(timeout=1)
        monitoring_threads.remove((thread, event))

def show_analytics():
    """Display email analytics"""
    try:
        if not active_manager or not active_manager.emails_data:
            console.print("[yellow]No email data available for analytics.[/]")
            return

        total_emails = len(active_manager.emails_data)
        active_emails = len([e for e in active_manager.emails_data if e.get('is_active', True)])
        
        table = Table(title="Email Analytics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Emails Generated", str(total_emails))
        table.add_row("Active Emails", str(active_emails))
        
        console.print(table)
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error showing analytics: {str(e)}")
        console.print("[red]Failed to show analytics. Check logs for details.[/]")

def export_data():
    """Export email data"""
    try:
        if not active_manager or not active_manager.emails_data:
            console.print("[yellow]No email data available to export.[/]")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tempmail_export_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(active_manager.emails_data, f, indent=2)
            
        console.print(f"[green]Data exported successfully to {filename}[/]")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        console.print("[red]Failed to export data. Check logs for details.[/]")

def generate_test_data():
    """Generate test data"""
    try:
        if not active_manager:
            console.print("[red]TempMail Manager not initialized.[/]")
            return
            
        num_emails = questionary.text(
            "How many test emails would you like to generate?",
            validate=lambda text: text.isdigit() and int(text) > 0,
            default="5"
        ).ask()
        
        for _ in range(int(num_emails)):
            email = f"test_{_}@example.com"
            active_manager.emails_data.append({
                'email': email,
                'provider': 'Test Provider',
                'created_at': datetime.now().isoformat(),
                'is_active': True
            })
        
        active_manager.save_data()
        console.print(f"[green]Successfully generated {num_emails} test emails[/]")
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error generating test data: {str(e)}")
        console.print("[red]Failed to generate test data. Check logs for details.[/]")

def forward_email_menu():
    """Handle email forwarding menu"""
    global active_manager
    
    # List available emails
    emails = [e['email'] for e in active_manager.emails_data]
    if not emails:
        console.print("[yellow]No emails available. Generate one first![/]")
        return
    
    # Select source email
    console.print("\n[bold cyan]Select source email:[/]")
    source_email = select_email_to_monitor()
    if not source_email:
        return
    
    # Get messages for source email
    messages = []
    for email_data in active_manager.emails_data:
        if email_data['email'] == source_email:
            messages = email_data['messages']
            break
    
    if not messages:
        console.print("[yellow]No messages available in source email![/]")
        return
    
    # Display messages
    table = Table(title=f"Messages in {source_email}")
    table.add_column("Index", style="cyan")
    table.add_column("Subject", style="white")
    table.add_column("From", style="green")
    table.add_column("Time", style="yellow")
    
    for idx, msg in enumerate(messages, 1):
        table.add_row(
            str(idx),
            msg['subject'],
            msg['sender'],
            msg['received_at']
        )
    
    console.print(table)
    
    # Select message
    choice = questionary.select(
        "Select message number",
        choices=[str(i) for i in range(1, len(messages) + 1)],
        use_indicator=True,
        use_shortcuts=True
    ).ask()
    message = messages[int(choice) - 1]
    
    # Select destination email
    console.print("\n[bold cyan]Select destination email:[/]")
    dest_email = select_email_to_monitor()
    if not dest_email:
        return
    
    # Forward the email
    with console.status("[bold green]Forwarding email...", spinner="dots"):
        success = active_manager.forward_email(
            from_email=source_email,
            to_email=dest_email,
            message_id=message['message_id']
        )
    
    if success:
        console.print("[green]Email forwarded successfully![/]")
    else:
        console.print("[red]Failed to forward email![/]")
    
    input("\nPress Enter to continue...")

def delete_email():
    """Delete one or more emails"""
    emails = active_manager.get_active_emails()
    
    if not emails:
        console.print("[yellow]No active emails to delete[/]")
        return
        
    # Display active emails
    table = Table(title="Active Emails")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Provider", style="blue")
    table.add_column("Created At", style="magenta")
    
    for i, email_data in enumerate(emails, 1):
        table.add_row(
            str(i),
            email_data['email'],
            email_data['provider'],
            email_data['created_at']
        )
    
    console.print(table)
    
    # Get user selection
    choice = Prompt.ask(
        "Enter email numbers to delete (comma-separated, e.g., 1,2,3)",
        default="1"
    )
    
    try:
        # Parse indices
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        emails_to_delete = []
        
        # Validate indices
        for idx in indices:
            if 0 <= idx < len(emails):
                emails_to_delete.append(emails[idx]['email'])
            else:
                console.print(f"[yellow]Invalid index: {idx + 1}[/]")
        
        if not emails_to_delete:
            console.print("[yellow]No valid emails selected[/]")
            return
            
        # Show emails to be deleted
        console.print("\nEmails to delete:")
        for email in emails_to_delete:
            console.print(f"- {email}")
            
        # Confirm deletion
        if Confirm.ask("\nAre you sure you want to delete these emails?"):
            deleted = []
            failed = []
            
            for email in emails_to_delete:
                if active_manager.delete_email(email):
                    deleted.append(email)
                else:
                    failed.append(email)
            
            if deleted:
                console.print(f"\n[green]Successfully deleted {len(deleted)} email(s)[/]")
            
            for email in failed:
                console.print(f"\n[red]Failed to delete {email}[/]")
                
            if not deleted and not failed:
                console.print("\n[yellow]No emails were deleted[/]")
        else:
            console.print("\n[yellow]Deletion cancelled[/]")
            
    except ValueError:
        console.print("[red]Please enter valid numbers[/]")
    
    input("\nPress Enter to continue...")

def list_active_emails():
    """List all active emails"""
    try:
        emails = active_manager.get_active_emails()
        
        if not emails:
            console.print("[yellow]No active emails[/yellow]")
            return
            
        table = Table(title="Active Emails")
        table.add_column("Index", justify="right", style="cyan")
        table.add_column("Email", style="green")
        table.add_column("Provider", style="blue")
        table.add_column("Created At", style="magenta")
        
        for i, email_data in enumerate(emails, 1):
            table.add_row(
                str(i),
                email_data['email'],
                email_data['provider'],
                email_data['created_at']
            )
        
        console.print(table)
        input("\nPress Enter to continue...")
        
    except Exception as e:
        logger.error(f"Error listing active emails: {str(e)}")
        console.print("[red]Error listing active emails[/red]")

def handle_menu_option(option: str):
    """Handle menu option selection"""
    try:
        if option == "1. Generate new email":
            generate_email()
        elif option == "2. Monitor emails":
            monitor_emails()
        elif option == "3. List active emails":
            list_active_emails()
        elif option == "4. Check messages":
            check_messages()
        elif option == "5. Forward email":
            forward_email_menu()
        elif option == "6. Export emails":
            export_data()
        elif option == "7. Delete email":
            delete_email()
        elif option == "8. Clear screen":
            clear_screen()
            display_welcome()
        elif option == "9. Exit":
            return True  # Signal to exit
    except Exception as e:
        logger.error(f"Error handling option {option}: {str(e)}")
        console.print(f"[red]Error:[/] {str(e)}")
    return False

def main():
    """Main application entry point"""
    try:
        # Display welcome message
        display_welcome()
        
        while True:
            try:
                # Display menu and get user choice
                option = display_menu()
                
                if not option:  # User pressed Ctrl+C or similar
                    break
                    
                # Handle the selected option
                if handle_menu_option(option) or option == "9. Exit":
                    break
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                console.print("[red]An error occurred. Please try again.[/red]")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        cleanup()  # This will handle the goodbye message

if __name__ == "__main__":
    main()
