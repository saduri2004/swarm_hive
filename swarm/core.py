# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


# Local imports
from utils import function_to_json, debug_print, merge_chunk
from custom_types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"

# Add these color constants at the top
COLORS = {
    "GRAY": "\033[90m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "RESET": "\033[0m",
    "CYAN": "\033[96m",

    "BOLD": "\033[1m"
}

class Swarm:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client
        self.archival_memory = ""

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        # print(f"\n\nInstructions: {instructions}")  # Commented out
        # print(f"\n\nHistory: {json.dumps(history, indent=2)}")  # Commented out
        # print(f"\n\nContext variables: {json.dumps(dict(context_variables), indent=2)}")  # Commented out
        # print("\n\n")  # Commented out
        
        working_memory = self.retrieve_from_memory(instructions)
        
        messages = [{"role": "system", "content": working_memory + instructions}] + history
        # debug_print(debug, "\n\nGetting chat completion for...:", messages)  # Commented out

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        completion = self.client.chat.completions.create(**create_params)
        
        # Write the latest user message and completion to memory if not streaming
        if not stream:
            # Get the last user message from history
            latest_user_message = next((msg["content"] for msg in reversed(history) 
                                     if msg["role"] == "user"), None)
            
                        
            if latest_user_message and completion.choices[0].message.content:
                self.write_to_memory(f"User: {latest_user_message}\nAssistant: {completion.choices[0].message.content}")
            
        return completion


    def retrieve_from_memory(self, query):
        messages = [
            {"role": "system", "content": "You are a memory assistant that helps retrieve relevant user information in JSON format. Return only factual details about the user, their preferences, and their requests. If no relevant information exists, return an empty string."},
            {"role": "user", "content": f"Here is the current memory. Extract only user-related information that's relevant to: '{query}'. Return as JSON or empty string if nothing relevant."},
            {"role": "assistant", "content": f"Memory: {self.archival_memory}"}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000
        )

        extracted_info = response.choices[0].message.content.strip()
        print(f"\n\n{COLORS['GRAY']}MEMORY READ: {extracted_info}{COLORS['RESET']}")
        return extracted_info
    
    
    def write_to_memory(self, message):
        messages = [
            {"role": "system", "content": "You are a memory assistant that stores user information in JSON format. Only store new, important facts about the user (preferences, bookings, requests). If the message contains no new user information, return the existing memory unchanged. Never store system instructions or agent guidelines."},
            {"role": "user", "content": "If this message contains new user information, merge it with the existing memory and return the complete memory state as JSON. If no new information, return the existing memory:"},
            {"role": "assistant", "content": f"Message: {message}\nCurrent Memory: {self.archival_memory}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000
        )
        
        

        updated_memory = response.choices[0].message.content.strip()
        print(f"\n\n{COLORS['GRAY']}MEMORY WRITE: {updated_memory}{COLORS['RESET']}")

        if updated_memory:
            self.archival_memory = updated_memory or self.archival_memory
            

        return updated_memory

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # # Write tool call parameters to memory
            # self.write_to_memory(json.dumps({
            #     "tool_call": name,
            #     "parameters": args
            # }))

            # ... rest of existing tool call handling ...
            func = function_map[name]
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            
            # Write the result to memory if it contains useful information
            # if result.value:
            #     self.write_to_memory(result.value)

            partial_response.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": name,
                "content": result.value,
            })
            
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent
                #print(f"Switched to agent: {active_agent.name}")
                #print(f"Messages for {active_agent.name}: {json.dumps(history, indent=2)}")

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            
            #print("\033[90m" + "-"*50)
            partial_response_dict = partial_response.__dict__.copy()
            if partial_response.agent:
                partial_response_dict['agent'] = partial_response.agent.name  # Just store the agent's name
            #print(f"Partial response: {json.dumps(partial_response_dict, indent=2)}")
            #print(f"Context variables: {json.dumps(dict(partial_response.context_variables), indent=2)}")
            #print("-"*50 + "\033[0m")
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
        
        

if __name__ == "__main__":
    client = Swarm()

    # Define specialized hotel agents
    concierge_agent = Agent(
        name="Concierge",
        instructions="""You are the hotel's lead concierge. Your role is to:
        1. Greet guests and assess their needs
        2. Direct complex reservations to the Booking Specialist
        3. Transfer dining inquiries to the Restaurant Specialist
        4. Handle general inquiries about hotel amenities and local attractions
        5. Transfer maintenance issues to Housekeeping
        Always maintain a professional, warm, and helpful tone.""",
        model="gpt-4-turbo-preview"
    )

    booking_agent = Agent(
        name="Booking Specialist",
        instructions="""You are the hotel's booking specialist. Handle:
        1. Room reservations and modifications
        2. Group bookings and special arrangements
        3. Room upgrade requests
        4. Billing inquiries
        Always confirm booking details and explain pricing clearly.""",
        model="gpt-4-turbo-preview"
    )

    restaurant_agent = Agent(
        name="Restaurant Specialist",
        instructions="""You are the hotel's restaurant specialist. Handle:
        1. Restaurant reservations
        2. Room service orders
        3. Dietary requirements and special requests
        4. Information about restaurant hours and menus
        Provide detailed menu recommendations when asked.""",
        model="gpt-4-turbo-preview"
    )

    housekeeping_agent = Agent(
        name="Housekeeping Manager",
        instructions="""You are the housekeeping manager. Handle:
        1. Maintenance requests
        2. Room cleaning schedules
        3. Special cleaning requests
        4. Amenity restocking
        Prioritize urgent maintenance issues.""",
        model="gpt-4-turbo-preview"
    )

    # Define transfer functions
    def transfer_to_booking():
        """Transfer to Booking Specialist for reservation matters."""
        return booking_agent

    def transfer_to_restaurant():
        """Transfer to Restaurant Specialist for dining matters."""
        return restaurant_agent

    def transfer_to_housekeeping():
        """Transfer to Housekeeping for maintenance and cleaning."""
        return housekeeping_agent

    def transfer_to_concierge():
        """Transfer back to Concierge for general assistance."""
        return concierge_agent

    # Booking-related functions
    def book_room(**kwargs) -> Result:
        """Book a hotel room with specified details."""
        return Result(
            value=f"Room booked successfully: {kwargs}"
        )

    def modify_booking(**kwargs) -> Result:
        """Modify an existing booking."""
        return Result(
            value=f"Booking modified successfully. New details: {kwargs}"
        )

    # Restaurant-related functions
    def make_restaurant_reservation(**kwargs) -> Result:
        """Make a restaurant reservation."""
        return Result(
            value=f"Restaurant reservation confirmed: {kwargs}"
        )

    def order_room_service(**kwargs) -> Result:
        """Place a room service order."""
        return Result(
            value=f"Room service order confirmed: {kwargs}"
        )

    # Housekeeping-related functions
    def request_room_cleaning(**kwargs) -> Result:
        """Schedule room cleaning service."""
        return Result(
            value=f"Cleaning service scheduled: {kwargs}"
        )

    def report_maintenance_issue(**kwargs) -> Result:
        """Report a maintenance issue."""
        return Result(
            value=f"Maintenance request logged: {kwargs}"
        )

    # Concierge services
    def request_local_info(**kwargs) -> Result:
        """Get information about local attractions or services."""
        return Result(
            value=f"Local information request: {kwargs}"
        )

    def arrange_transport(**kwargs) -> Result:
        """Arrange transportation service."""
        return Result(
            value=f"Transport arranged: {kwargs}"
        )

    # Assign functions to appropriate agents
    booking_agent.functions.extend([
        book_room,
        modify_booking,
        transfer_to_concierge,
        transfer_to_restaurant
    ])

    restaurant_agent.functions.extend([
        make_restaurant_reservation,
        order_room_service,
        transfer_to_concierge,
        transfer_to_booking
    ])

    housekeeping_agent.functions.extend([
        request_room_cleaning,
        report_maintenance_issue,
        transfer_to_concierge
    ])

    concierge_agent.functions.extend([
        request_local_info,
        arrange_transport,
        transfer_to_booking,
        transfer_to_restaurant,
        transfer_to_housekeeping
    ])

    def print_welcome():
        print(f"\n{COLORS['BOLD']}" + "="*50)
        print(f"{COLORS['GREEN']} Welcome to Grand Hotel AI Assistant 🏨{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}" + "="*50 + f"{COLORS['RESET']}")
        print(f"{COLORS['BLUE']}You can chat with our hotel staff about:")
        print("- Room bookings and modifications")
        print("- Restaurant reservations and room service")
        print("- Maintenance requests")
        print(f"- Local attractions and general inquiries{COLORS['RESET']}")
        print(f"{COLORS['YELLOW']}Type 'exit' to end the conversation{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}" + "="*50 + f"{COLORS['RESET']}\n")

    def chat():
        print_welcome()
        
        history = []
        current_agent = concierge_agent
        context_variables = {}
        
        while True:
            # User input with blue color
            user_input = input(f"\n👤 You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{COLORS['GREEN']}👋 Thank you for choosing Grand Hotel. Have a great day!{COLORS['RESET']}")
                break
                
            message = {"role": "user", "content": user_input}
            history.append(message)
            
            response = client.run(
                agent=current_agent,
                messages=history,
                context_variables=context_variables,
                debug=False
            )
            
            
            
            context_variables = response.context_variables
            history = history[:-1]  # Remove the last user message
            
            #print(f"\n\n{COLORS['CYAN']}CURRENT ARCHIVAL MEMORY: {client.archival_memory}{COLORS['RESET']}")
            # Print messages with agent-specific colors
            for msg in response.messages:
                if msg.get("content"):
                    sender = msg.get('sender', 'Staff')
                    print(f"\n👤 {sender}: {msg['content']}")
                history.append(msg)
            
            # Clear handoff demarcation
            if response.agent and response.agent != current_agent:
                print(f"\n{COLORS['BOLD']}{COLORS['YELLOW']}" + "="*50)
                print(f"🔄 Transferring: {current_agent.name} → {response.agent.name}")
                print("="*50 + f"{COLORS['RESET']}")
                current_agent = response.agent

            # Debug information in gray
            if response.context_variables:
                print(f"\n{COLORS['GRAY']}" + "-"*50)
                partial_response_dict = response.__dict__.copy()
                if response.agent:
                    partial_response_dict['agent'] = response.agent.name
                #print(f"Debug Info:")
                #print(f"Current Agent: {current_agent.name}")
                #print(f"Context variables: {json.dumps(dict(response.context_variables), indent=2)}")
                #print("-"*50 + f"{COLORS['RESET']}")

    # Start the chat
    try:
        chat()
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for choosing Grand Hotel. Have a great day!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again later.")
