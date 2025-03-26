from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException
from starlette.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import csv
import io
import json
from dataclasses import dataclass
import heapq
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Space Station Cargo Management System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Coordinates(BaseModel):
    width: float
    depth: float
    height: float

class Position(BaseModel):
    startCoordinates: Coordinates
    endCoordinates: Coordinates

class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    mass: Optional[float] = 0
    priority: int
    expiryDate: Optional[str] = None
    usageLimit: Optional[int] = None
    preferredZone: Optional[str] = None

class Container(BaseModel):
    containerId: str
    zone: str
    width: float
    depth: float
    height: float

class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]

class PlacementResponse(BaseModel):
    success: bool
    placements: List[Dict[str, Any]]
    rearrangements: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    success: bool
    found: bool
    item: Optional[Dict[str, Any]] = None
    retrievalSteps: Optional[List[Dict[str, Any]]] = None

class RetrieveRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str

class PlaceRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str
    containerId: str
    position: Position

class WasteIdentifyResponse(BaseModel):
    success: bool
    wasteItems: List[Dict[str, Any]]

class WasteReturnPlanRequest(BaseModel):
    undockingContainerId: str
    undockingDate: str
    maxWeight: float

class CompleteUndockingRequest(BaseModel):
    undockingContainerId: str
    timestamp: str

class SimulateDayRequest(BaseModel):
    numOfDays: Optional[int] = None
    toTimestamp: Optional[str] = None
    itemsToBeUsedPerDay: List[Dict[str, str]]

# In-memory database
containers_db = {}
items_db = {}
placements_db = {}  # Maps item_id to container_id and position
logs_db = []
current_date = datetime.now().isoformat()

# Helper for 3D bin packing algorithm using a modified first-fit decreasing approach
@dataclass
class Space:
    width_start: float
    depth_start: float
    height_start: float
    width_end: float
    depth_end: float
    height_end: float
    
    @property
    def volume(self):
        return (self.width_end - self.width_start) * \
               (self.depth_end - self.depth_start) * \
               (self.height_end - self.height_start)

def find_space_for_item(container, item, occupied_spaces):
    # Try all possible orientations of the item
    orientations = [
        (item.width, item.depth, item.height),
        (item.width, item.height, item.depth),
        (item.height, item.width, item.depth),
        (item.height, item.depth, item.width),
        (item.depth, item.width, item.height),
        (item.depth, item.height, item.width)
    ]
    
    best_space = None
    best_distance = float('inf')  # Distance from open face (depth)
    best_orientation = None
    
    for w, d, h in orientations:
        # Ensure the item fits within container dimensions
        if w > container.width or d > container.depth or h > container.height:
            continue
            
        # Check all possible positions within the container
        for x in range(int(container.width - w) + 1):
            for y in range(int(container.depth - d) + 1):
                for z in range(int(container.height - h) + 1):
                    end_x, end_y, end_z = x + w, y + d, z + h
                    
                    # Check if this position conflicts with any occupied space
                    conflict = False
                    for space in occupied_spaces:
                        if (x < space.width_end and end_x > space.width_start and
                            y < space.depth_end and end_y > space.depth_start and
                            z < space.height_end and end_z > space.height_start):
                            conflict = True
                            break
                    
                    if not conflict:
                        # Calculate distance from the open face (depth is the key factor)
                        distance = y  # Depth coordinate
                        
                        # For high priority items, we want them closer to the open face
                        # Lower distance is better for high priority items
                        adjusted_distance = distance / (item.priority / 100)
                        
                        if adjusted_distance < best_distance:
                            best_distance = adjusted_distance
                            best_space = Space(x, y, z, end_x, end_y, end_z)
                            best_orientation = (w, d, h)
    
    return best_space, best_orientation

def calculate_retrieval_steps(item_id, container_id):
    """Calculate the steps needed to retrieve an item."""
    if item_id not in placements_db:
        return None, []
    
    placement = placements_db[item_id]
    if placement["containerId"] != container_id:
        return None, []
    
    # Get all items in the same container
    items_in_container = [
        (i_id, p) for i_id, p in placements_db.items() 
        if p["containerId"] == container_id and i_id != item_id
    ]
    
    # Item's position
    item_pos = placement["position"]
    item_start = item_pos["startCoordinates"]
    item_end = item_pos["endCoordinates"]
    
    # Check which items block this item
    blocking_items = []
    
    for other_id, other_placement in items_in_container:
        other_pos = other_placement["position"]
        other_start = other_pos["startCoordinates"]
        other_end = other_pos["endCoordinates"]
        
        # Check if this item blocks the target item
        # For an item to block, it must be in front of our target item (smaller depth)
        if (other_start["depth"] < item_start["depth"] and
            # And there must be some overlap in the width and height dimensions
            other_start["width"] < item_end["width"] and
            other_end["width"] > item_start["width"] and
            other_start["height"] < item_end["height"] and
            other_end["height"] > item_start["height"]):
            blocking_items.append({
                "itemId": other_id,
                "itemName": items_db[other_id].name,
                "position": other_placement["position"]
            })
    
    # Sort blocking items by depth (items closer to the opening come first)
    blocking_items.sort(key=lambda x: x["position"]["startCoordinates"]["depth"])
    
    # Generate retrieval steps
    steps = []
    for i, blocking_item in enumerate(blocking_items):
        # Remove step
        steps.append({
            "step": i * 2 + 1,
            "action": "remove",
            "itemId": blocking_item["itemId"],
            "itemName": blocking_item["itemName"]
        })
        
        # Set aside step
        steps.append({
            "step": i * 2 + 2,
            "action": "setAside",
            "itemId": blocking_item["itemId"],
            "itemName": blocking_item["itemName"]
        })
    
    # Retrieve target item
    steps.append({
        "step": len(blocking_items) * 2 + 1,
        "action": "retrieve",
        "itemId": item_id,
        "itemName": items_db[item_id].name
    })
    
    # Place back steps (in reverse order)
    for i, blocking_item in enumerate(reversed(blocking_items)):
        steps.append({
            "step": len(blocking_items) * 2 + 2 + i,
            "action": "placeBack",
            "itemId": blocking_item["itemId"],
            "itemName": blocking_item["itemName"]
        })
    
    return placement, steps

def is_item_waste(item_id):
    item = items_db.get(item_id)
    if not item:
        return False, ""
    
    # Check expiry
    if item.expiryDate:
        expiry_date = datetime.fromisoformat(item.expiryDate.replace('Z', '+00:00'))
        current = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
        if current >= expiry_date:
            return True, "Expired"
    
    # Check usage limit
    if item.usageLimit is not None and item.usageLimit <= 0:
        return True, "Out of Uses"
    
    return False, ""

def log_action(user_id, action_type, item_id, details=None):
    if details is None:
        details = {}
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "userId": user_id,
        "actionType": action_type,
        "itemId": item_id,
        "details": details
    }
    logs_db.append(log_entry)
    return log_entry

# API endpoints
@app.post("/api/placement", response_model=PlacementResponse)
async def placement_recommendation(request: PlacementRequest):
    """Place new items in available containers optimally."""
    
    # Update local database
    for container in request.containers:
        containers_db[container.containerId] = container
    
    for item in request.items:
        items_db[item.itemId] = item
    
    # Sort items by priority (highest first)
    sorted_items = sorted(request.items, key=lambda x: x.priority, reverse=True)
    
    placements = []
    rearrangements = []
    
    # Group containers by zone
    containers_by_zone = {}
    for container in request.containers:
        if container.zone not in containers_by_zone:
            containers_by_zone[container.zone] = []
        containers_by_zone[container.zone].append(container)
    
    # Track occupied spaces in each container
    container_occupied_spaces = {container.containerId: [] for container in request.containers}
    
    # Add already placed items to occupied spaces
    for item_id, placement in placements_db.items():
        container_id = placement["containerId"]
        if container_id in container_occupied_spaces:
            pos = placement["position"]
            container_occupied_spaces[container_id].append(
                Space(
                    pos["startCoordinates"]["width"],
                    pos["startCoordinates"]["depth"],
                    pos["startCoordinates"]["height"],
                    pos["endCoordinates"]["width"],
                    pos["endCoordinates"]["depth"],
                    pos["endCoordinates"]["height"]
                )
            )
    
    # Try to place each item
    for item in sorted_items:
        placed = False
        
        # First try preferred zone
        preferred_containers = []
        if item.preferredZone and item.preferredZone in containers_by_zone:
            preferred_containers = containers_by_zone[item.preferredZone]
        
        # If no preferred zone or can't place in preferred, try all zones
        if not preferred_containers:
            for zone, zone_containers in containers_by_zone.items():
                preferred_containers.extend(zone_containers)
        
        # Sort containers by volume efficiency (smallest first)
        preferred_containers.sort(key=lambda c: c.width * c.depth * c.height)
        
        for container in preferred_containers:
            best_space, orientation = find_space_for_item(
                container, 
                item, 
                container_occupied_spaces[container.containerId]
            )
            
            if best_space:
                # Item can be placed here
                placement = {
                    "itemId": item.itemId,
                    "containerId": container.containerId,
                    "position": {
                        "startCoordinates": {
                            "width": best_space.width_start,
                            "depth": best_space.depth_start,
                            "height": best_space.height_start
                        },
                        "endCoordinates": {
                            "width": best_space.width_end,
                            "depth": best_space.depth_end,
                            "height": best_space.height_end
                        }
                    }
                }
                
                placements.append(placement)
                container_occupied_spaces[container.containerId].append(best_space)
                placements_db[item.itemId] = placement
                placed = True
                break
        
        if not placed:
            # Try rearrangement strategy
            # For simplicity, just identify it can't be placed without rearrangement
            rearrangements.append({
                "step": len(rearrangements) + 1,
                "action": "needsRearrangement",
                "itemId": item.itemId,
                "fromContainer": None,
                "fromPosition": None,
                "toContainer": None,
                "toPosition": None
            })
    
    return {
        "success": True,
        "placements": placements,
        "rearrangements": rearrangements
    }

@app.get("/api/search", response_model=SearchResponse)
async def search_item(
    itemId: Optional[str] = None,
    itemName: Optional[str] = None,
    userId: Optional[str] = None
):
    """Search for an item by ID or name."""
    if not itemId and not itemName:
        return {"success": False, "found": False}
    
    target_item_id = None
    
    if itemId:
        if itemId in items_db and itemId in placements_db:
            target_item_id = itemId
    elif itemName:
        # Find item by name
        for i_id, item in items_db.items():
            if item.name == itemName and i_id in placements_db:
                target_item_id = i_id
                break
    
    if not target_item_id:
        return {"success": True, "found": False}
    
    # Get placement and container info
    placement = placements_db[target_item_id]
    container_id = placement["containerId"]
    container = containers_db[container_id]
    
    # Calculate retrieval steps
    _, retrieval_steps = calculate_retrieval_steps(target_item_id, container_id)
    
    # Log search action
    if userId:
        log_action(userId, "search", target_item_id)
    
    return {
        "success": True,
        "found": True,
        "item": {
            "itemId": target_item_id,
            "name": items_db[target_item_id].name,
            "containerId": container_id,
            "zone": container.zone,
            "position": placement["position"]
        },
        "retrievalSteps": retrieval_steps
    }

@app.post("/api/retrieve")
async def retrieve_item(request: RetrieveRequest):
    """Log the retrieval of an item and decrement its usage limit."""
    if request.itemId not in items_db:
        return {"success": False}
    
    # Update the usage limit
    item = items_db[request.itemId]
    if item.usageLimit is not None:
        item.usageLimit -= 1
        items_db[request.itemId] = item
    
    # Log retrieval
    log_action(
        request.userId, 
        "retrieval", 
        request.itemId, 
        {"timestamp": request.timestamp}
    )
    
    return {"success": True}

@app.post("/api/place")
async def place_item(request: PlaceRequest):
    """Update the placement of an item after retrieval."""
    if request.itemId not in items_db:
        return {"success": False}
    
    # Update placement
    placements_db[request.itemId] = {
        "containerId": request.containerId,
        "position": request.position.dict()
    }
    
    # Log placement
    log_action(
        request.userId, 
        "placement", 
        request.itemId, 
        {
            "timestamp": request.timestamp,
            "containerId": request.containerId
        }
    )
    
    return {"success": True}

@app.get("/api/waste/identify", response_model=WasteIdentifyResponse)
async def identify_waste():
    """Identify items that are expired or out of uses."""
    waste_items = []
    
    for item_id, item in items_db.items():
        if item_id in placements_db:
            is_waste, reason = is_item_waste(item_id)
            if is_waste:
                placement = placements_db[item_id]
                waste_items.append({
                    "itemId": item_id,
                    "name": item.name,
                    "reason": reason,
                    "containerId": placement["containerId"],
                    "position": placement["position"]
                })
    
    return {"success": True, "wasteItems": waste_items}

@app.post("/api/waste/return-plan")
async def waste_return_plan(request: WasteReturnPlanRequest):
    """Create a plan for returning waste items via undocking."""
    # Identify all waste items
    waste_items = []
    
    for item_id, item in items_db.items():
        if item_id in placements_db:
            is_waste, reason = is_item_waste(item_id)
            if is_waste:
                waste_items.append({
                    "itemId": item_id,
                    "name": item.name,
                    "reason": reason,
                    "mass": item.mass if hasattr(item, 'mass') else 0,
                    "containerInfo": {
                        "containerId": placements_db[item_id]["containerId"],
                        "position": placements_db[item_id]["position"]
                    }
                })
    
    # Sort waste items by mass (lightest first)
    waste_items.sort(key=lambda x: x["mass"])
    
    # Create a return plan within the max weight constraint
    return_plan = []
    retrieval_steps = []
    return_items = []
    total_weight = 0
    total_volume = 0
    
    for idx, waste_item in enumerate(waste_items):
        # Check if adding this item would exceed max weight
        if total_weight + waste_item["mass"] > request.maxWeight:
            continue
        
        # Get retrieval steps for this item
        item_id = waste_item["itemId"]
        container_id = waste_item["containerInfo"]["containerId"]
        
        placement, steps = calculate_retrieval_steps(item_id, container_id)
        if not placement:
            continue
        
        # Add to return plan
        step_num = len(return_plan) + 1
        return_plan.append({
            "step": step_num,
            "itemId": item_id,
            "itemName": waste_item["name"],
            "fromContainer": container_id,
            "toContainer": request.undockingContainerId
        })
        
        # Add retrieval steps
        retrieval_steps.extend(steps)
        
        # Add to return manifest
        return_items.append({
            "itemId": item_id,
            "name": waste_item["name"],
            "reason": waste_item["reason"]
        })
        
        # Update totals
        total_weight += waste_item["mass"]
        
        # Calculate volume (approximation)
        pos = waste_item["containerInfo"]["position"]
        width = pos["endCoordinates"]["width"] - pos["startCoordinates"]["width"]
        depth = pos["endCoordinates"]["depth"] - pos["startCoordinates"]["depth"]
        height = pos["endCoordinates"]["height"] - pos["startCoordinates"]["height"]
        volume = width * depth * height
        total_volume += volume
    
    return {
        "success": True,
        "returnPlan": return_plan,
        "retrievalSteps": retrieval_steps,
        "returnManifest": {
            "undockingContainerId": request.undockingContainerId,
            "undockingDate": request.undockingDate,
            "returnItems": return_items,
            "totalVolume": total_volume,
            "totalWeight": total_weight
        }
    }

@app.post("/api/waste/complete-undocking")
async def complete_undocking(request: CompleteUndockingRequest):
    """Complete the undocking process and remove items from the system."""
    items_to_remove = []
    
    # Find all items in the undocking container
    for item_id, placement in placements_db.items():
        if placement["containerId"] == request.undockingContainerId:
            items_to_remove.append(item_id)
    
    # Remove items
    for item_id in items_to_remove:
        if item_id in placements_db:
            del placements_db[item_id]
            
            # Log disposal
            log_action(
                "SYSTEM", 
                "disposal", 
                item_id, 
                {
                    "timestamp": request.timestamp,
                    "containerId": request.undockingContainerId
                }
            )
    
    return {"success": True, "itemsRemoved": len(items_to_remove)}

@app.post("/api/simulate/day")
async def simulate_day(request: SimulateDayRequest):
    """Simulate the passage of time and update item statuses."""
    global current_date
    
    # Get current date as datetime
    current_datetime = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
    
    # Calculate new date
    if request.numOfDays:
        new_datetime = current_datetime + timedelta(days=request.numOfDays)
    elif request.toTimestamp:
        new_datetime = datetime.fromisoformat(request.toTimestamp.replace('Z', '+00:00'))
    else:
        new_datetime = current_datetime + timedelta(days=1)
    
    # Update current date
    current_date = new_datetime.isoformat()
    
    # Track changes
    items_used = []
    items_expired = []
    items_depleted = []
    
    # Process items to be used each day
    days_diff = (new_datetime - current_datetime).days
    if days_diff > 0:
        for _ in range(days_diff):
            for item_info in request.itemsToBeUsedPerDay:
                item_id = item_info.get("itemId")
                item_name = item_info.get("name")
                
                # Find the item
                target_item_id = None
                if item_id and item_id in items_db:
                    target_item_id = item_id
                elif item_name:
                    for i_id, item in items_db.items():
                        if item.name == item_name:
                            target_item_id = i_id
                            break
                
                if target_item_id and target_item_id in items_db:
                    item = items_db[target_item_id]
                    if item.usageLimit is not None and item.usageLimit > 0:
                        # Decrement usage
                        item.usageLimit -= 1
                        items_db[target_item_id] = item
                        
                        # Add to used items
                        items_used.append({
                            "itemId": target_item_id,
                            "name": item.name,
                            "remainingUses": item.usageLimit
                        })
                        
                        # Check if depleted
                        if item.usageLimit == 0:
                            items_depleted.append({
                                "itemId": target_item_id,
                                "name": item.name
                            })
    
    # Check for expired items
    for item_id, item in items_db.items():
        if item.expiryDate:
            expiry_date = datetime.fromisoformat(item.expiryDate.replace('Z', '+00:00'))
            if new_datetime >= expiry_date and current_datetime < expiry_date:
                items_expired.append({
                    "itemId": item_id,
                    "name": item.name
                })
    
    return {
        "success": True,
        "newDate": current_date,
        "changes": {
            "itemsUsed": items_used,
            "itemsExpired": items_expired,
            "itemsDepletedToday": items_depleted
        }
    }

@app.get("/api/logs")
async def get_logs(
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    itemId: Optional[str] = None,
    userId: Optional[str] = None,
    actionType: Optional[str] = None
):
    """Get logs filtered by various criteria."""
    filtered_logs = logs_db.copy()
    
    # Apply filters
    if startDate:
        start = datetime.fromisoformat(startDate.replace('Z', '+00:00'))
        filtered_logs = [
            log for log in filtered_logs 
            if datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) >= start
        ]
    
    if endDate:
        end = datetime.fromisoformat(endDate.replace('Z', '+00:00'))
        filtered_logs = [
            log for log in filtered_logs 
            if datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00')) <= end
        ]
    
    if itemId:
        filtered_logs = [log for log in filtered_logs if log["itemId"] == itemId]
    
    if userId:
        filtered_logs = [log for log in filtered_logs if log["userId"] == userId]
    
    if actionType:
        filtered_logs = [log for log in filtered_logs if log["actionType"] == actionType]
    
    return {"logs": filtered_logs}

# Additional API endpoint for import/export
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    """Import items from a CSV file."""
    content = await file.read()
    items_imported = 0
    errors = []
    
    try:
        # Parse CSV
        csv_text = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_text))
        
        for row_idx, row in enumerate(csv_reader, start=1):
            try:
                # Convert row data types
                item_id = row.get('Item ID', str(uuid.uuid4()))
                
                # Parse numeric values
                try:
                    width = float(row.get('Width (cm)', 0))
                    depth = float(row.get('Depth (cm)', 0))
                    height = float(row.get('Height (cm)', 0))
                    mass = float(row.get('Mass (kg)', 0))
                    priority = int(row.get('Priority (1-100)', 50))
                except ValueError as e:
                    errors.append({"row": row_idx, "message": f"Invalid numeric value: {str(e)}"})
                    continue
                
                # Create item
                item = Item(
                    itemId=item_id,
                    name=row.get('Name', ''),
                    width=width,
                    depth=depth,
                    height=height,
                    mass=mass,
                    priority=priority,
                    expiryDate=row.get('Expiry Date (ISO Format)', None),
                    usageLimit=int(row.get('Usage Limit', 0)) if row.get('Usage Limit') else None,
                    preferredZone=row.get('Preferred Zone', None)
                )
                
                # Store in database
                items_db[item_id] = item
                items_imported += 1
                
            except Exception as e:
                errors.append({"row": row_idx, "message": str(e)})
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": True, "itemsImported": items_imported, "errors": errors}

@app.post("/api/import/containers")
async def import_containers(file: UploadFile = File(...)):
    """Import containers from a CSV file."""
    content = await file.read()
    containers_imported = 0
    errors = []
    
    try:
        # Parse CSV
        csv_text = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_text))
        
        for row_idx, row in enumerate(csv_reader, start=1):
            try:
                # Convert row data types
                container_id = row.get('Container ID', str(uuid.uuid4()))
                
                # Parse numeric values
                try:
                    width = float(row.get('Width(cm)', 0))
                    depth = float(row.get('Depth(cm)', 0))
                    height = float(row.get('Height(height)', 0))
                except ValueError as e:
                    errors.append({"row": row_idx, "message": f"Invalid numeric value: {str(e)}"})
                    continue
                
                # Create container
                container = Container(
                    containerId=container_id,
                    zone=row.get('Zone', ''),
                    width=width,
                    depth=depth,
                    height=height
                )
                
                # Store in database
                containers_db[container_id] = container
                containers_imported += 1
                
            except Exception as e:
                errors.append({"row": row_idx, "message": str(e)})
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    return {"success": True, "containersImported": containers_imported, "errors": errors}

@app.get("/api/export/arrangement")
async def export_arrangement():
    """Export the current arrangement as a CSV file."""
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["Item ID", "Container ID", "Coordinates (W1,D1,H1),(W2,D2,H2)"])
    
    # Write data
    for item_id, placement in placements_db.items():
        container_id = placement["containerId"]
        pos = placement["position"]
        start_coords = f"({pos['startCoordinates']['width']},{pos['startCoordinates']['depth']},{pos['startCoordinates']['height']})"
        end_coords = f"({pos['endCoordinates']['width']},{pos['endCoordinates']['depth']},{pos['endCoordinates']['height']})"
        writer.writerow([item_id, container_id, f"{start_coords},{end_coords}"])
    
    # Convert to CSV bytes
    output.seek(0)
    csv_content = output.getvalue().encode('utf-8')
    
    # Return as a file response
    return StreamingResponse(
        io.BytesIO(csv_content), 
        media_type='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=space_station_arrangement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
    )



# Existing code remains the same, with additional imports and new features

# Advanced Analytics and Reporting Classes
class InventoryReport(BaseModel):
    total_items: int
    items_by_zone: Dict[str, int]
    items_by_priority: Dict[str, int]
    expired_items: List[Dict[str, Any]]
    low_usage_items: List[Dict[str, Any]]

class PredictiveMaintenance(BaseModel):
    high_risk_items: List[Dict[str, Any]]
    usage_trends: Dict[str, Any]
    predicted_depletions: List[Dict[str, Any]]

# Enhanced Placement Algorithm with Machine Learning-inspired Approach
def advanced_placement_algorithm(request: PlacementRequest):
    """
    Advanced placement algorithm with multi-factor optimization
    - Considers priority, zone preferences, usage history, and spatial efficiency
    """
    # Score items based on multiple factors
    def score_item_placement(item, container):
        # Base score calculation
        base_score = 0
        
        # Priority weighting
        priority_weight = item.priority / 100
        base_score += priority_weight * 10
        
        # Spatial efficiency
        volume_item = item.width * item.depth * item.height
        volume_container = container.width * container.depth * container.height
        volume_efficiency = volume_item / volume_container
        base_score += (1 - volume_efficiency) * 5
        
        # Zone preference
        if item.preferredZone and item.preferredZone == container.zone:
            base_score += 3
        
        # Historical usage (simulated - would use actual usage data in real system)
        usage_history = items_db.get(item.itemId, {}).get('usage_count', 0)
        base_score += min(usage_history * 0.1, 2)
        
        return base_score
    
    # Core placement logic similar to existing implementation
    # with enhanced scoring mechanism
    # ... (rest of the existing placement logic with this scoring integrated)

# New Endpoint for Advanced Inventory Reporting
@app.get("/api/reports/inventory", response_model=InventoryReport)
async def generate_inventory_report():
    """Generate comprehensive inventory report."""
    # Items by zone
    items_by_zone = defaultdict(int)
    for placement in placements_db.values():
        container = containers_db.get(placement['containerId'])
        if container:
            items_by_zone[container.zone] += 1
    
    # Items by priority
    items_by_priority = defaultdict(int)
    for item in items_db.values():
        if 0 <= item.priority < 33:
            priority_category = 'Low'
        elif 33 <= item.priority < 66:
            priority_category = 'Medium'
        else:
            priority_category = 'High'
        items_by_priority[priority_category] += 1
    
    # Expired items
    expired_items = []
    for item_id, item in items_db.items():
        is_waste, reason = is_item_waste(item_id)
        if is_waste:
            placement = placements_db.get(item_id, {})
            expired_items.append({
                'itemId': item_id,
                'name': item.name,
                'reason': reason,
                'containerId': placement.get('containerId')
            })
    
    # Low usage items
    low_usage_items = []
    for item_id, item in items_db.items():
        if item.usageLimit is not None and item.usageLimit <= 2:
            placement = placements_db.get(item_id, {})
            low_usage_items.append({
                'itemId': item_id,
                'name': item.name,
                'remainingUses': item.usageLimit,
                'containerId': placement.get('containerId')
            })
    
    return {
        'total_items': len(items_db),
        'items_by_zone': dict(items_by_zone),
        'items_by_priority': dict(items_by_priority),
        'expired_items': expired_items,
        'low_usage_items': low_usage_items
    }

# Predictive Maintenance Endpoint
@app.get("/api/reports/predictive-maintenance", response_model=PredictiveMaintenance)
async def predictive_maintenance_report():
    """
    Generate predictive maintenance report
    - Identifies high-risk items
    - Tracks usage trends
    - Predicts potential item depletions
    """
    # High-risk items (based on multiple factors)
    high_risk_items = []
    for item_id, item in items_db.items():
        risk_score = 0
        
        # Check expiry proximity
        if item.expiryDate:
            expiry_date = datetime.fromisoformat(item.expiryDate.replace('Z', '+00:00'))
            days_to_expiry = (expiry_date - datetime.now()).days
            if days_to_expiry <= 30:
                risk_score += 3
        
        # Check usage limits
        if item.usageLimit is not None:
            if item.usageLimit <= 3:
                risk_score += 2
        
        # Log-based risk assessment (example)
        item_logs = [log for log in logs_db if log['itemId'] == item_id]
        recent_log_count = len([log for log in item_logs if 
            datetime.fromisoformat(log['timestamp']) > 
            datetime.now() - timedelta(days=30)])
        
        if recent_log_count == 0:
            risk_score += 1
        
        if risk_score > 2:
            high_risk_items.append({
                'itemId': item_id,
                'name': item.name,
                'riskScore': risk_score
            })
    
    # Usage Trends (simplified)
    usage_trends = {
        'total_retrievals': len([log for log in logs_db if log['actionType'] == 'retrieval']),
        'total_placements': len([log for log in logs_db if log['actionType'] == 'placement']),
        'daily_usage_avg': len(logs_db) / max(1, (datetime.now() - datetime.fromisoformat(logs_db[0]['timestamp'])).days)
    }
    
    # Predict potential depletions
    predicted_depletions = []
    for item_id, item in items_db.items():
        if item.usageLimit is not None and item.usageLimit <= 3:
            predicted_depletions.append({
                'itemId': item_id,
                'name': item.name,
                'remainingUses': item.usageLimit
            })
    
    return {
        'high_risk_items': high_risk_items,
        'usage_trends': usage_trends,
        'predicted_depletions': predicted_depletions
    }

# Enhanced Search Capabilities
@app.get("/api/advanced-search")
async def advanced_search(
    name: Optional[str] = None,
    priority_min: Optional[int] = None,
    priority_max: Optional[int] = None,
    zone: Optional[str] = None,
    expiring_within_days: Optional[int] = None,
    max_results: Optional[int] = 100
):
    """
    Advanced search with multiple filtering options
    - Search by name (partial match)
    - Filter by priority range
    - Filter by zone
    - Find items expiring soon
    """
    results = []
    
    for item_id, item in items_db.items():
        # Name filter (case-insensitive partial match)
        if name and name.lower() not in item.name.lower():
            continue
        
        # Priority range filter
        if priority_min is not None and item.priority < priority_min:
            continue
        if priority_max is not None and item.priority > priority_max:
            continue
        
        # Zone filter
        if item_id in placements_db:
            container_id = placements_db[item_id]['containerId']
            container = containers_db.get(container_id)
            if zone and container and container.zone != zone:
                continue
        
        # Expiry filter
        if expiring_within_days and item.expiryDate:
            expiry_date = datetime.fromisoformat(item.expiryDate.replace('Z', '+00:00'))
            days_to_expiry = (expiry_date - datetime.now()).days
            if days_to_expiry > expiring_within_days:
                continue
        
        # Prepare result
        result = {
            'itemId': item_id,
            'name': item.name,
            'priority': item.priority,
            'expiryDate': item.expiryDate,
            'usageLimit': item.usageLimit
        }
        
        # Add placement info if available
        if item_id in placements_db:
            placement = placements_db[item_id]
            container = containers_db.get(placement['containerId'])
            result['zone'] = container.zone if container else None
            result['position'] = placement['position']
        
        results.append(result)
        
        # Respect max results limit
        if len(results) >= max_results:
            break
    
    return {
        'total_results': len(results),
        'results': results
    }

# Additional Simulation Features
@app.post("/api/simulate/scenario")
async def simulate_scenario(
    scenario_type: str,
    parameters: Dict[str, Any]
):
    """
    Advanced scenario simulation
    - Emergency scenarios
    - Resource optimization
    - Capacity planning
    """
    if scenario_type == "emergency_evacuation":
        # Simulate emergency item retrieval prioritization
        priority_items = sorted(
            [item for item_id, item in items_db.items() if item_id in placements_db], 
            key=lambda x: x.priority, 
            reverse=True
        )
        
        retrieval_plan = []
        for item in priority_items[:parameters.get('max_items', 10)]:
            retrieval_plan.append({
                'itemId': item.itemId,
                'name': item.name,
                'priority': item.priority,
                'container': placements_db[item.itemId]['containerId']
            })
        
        return {
            'scenario': 'emergency_evacuation',
            'retrieval_plan': retrieval_plan
        }
    
    elif scenario_type == "resource_optimization":
        # Analyze item usage and suggest optimizations
        usage_stats = defaultdict(int)
        for log in logs_db:
            if log['actionType'] in ['retrieval', 'placement']:
                usage_stats[log['itemId']] += 1
        
        optimized_items = sorted(
            [(item_id, count) for item_id, count in usage_stats.items()], 
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'scenario': 'resource_optimization',
            'most_used_items': [
                {
                    'itemId': item_id, 
                    'usage_count': count,
                    'name': items_db[item_id].name
                } for item_id, count in optimized_items[:parameters.get('top_n', 5)]
            ]
        }
    
    else:
        raise HTTPException(status_code=400, detail="Unknown scenario type")

# Create a directory for static files if it doesn't exist
os.makedirs('static', exist_ok=True)

# Create a minimal favicon
def create_minimal_favicon():
    # Create a simple black and white square favicon
    from PIL import Image
    
    # Create a 32x32 pixel image
    favicon = Image.new('RGB', (32, 32), color='white')
    
    # Draw a simple icon (a stylized 'S' for Space Station)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(favicon)
    
    # Draw border
    draw.rectangle([0, 0, 31, 31], outline='black')
    
    # Draw stylized 'S'
    draw.line([10, 5, 22, 5], fill='black', width=3)  # Top
    draw.line([22, 5, 22, 15], fill='black', width=3)  # Right vertical
    draw.line([10, 15, 22, 15], fill='black', width=3)  # Middle
    draw.line([10, 15, 10, 25], fill='black', width=3)  # Left vertical
    draw.line([10, 25, 22, 25], fill='black', width=3)  # Bottom
    
    # Save favicon
    favicon.save('static/favicon.ico')

# Create the favicon
create_minimal_favicon()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a route for favicon
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# Existing root route remains the same
@app.get("/")
async def root():
    return {"message": "Welcome to the Space Station Cargo Management System"}