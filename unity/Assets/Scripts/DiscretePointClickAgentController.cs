using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace UnityStandardAssets.Characters.FirstPerson
{
[Serializable]
public class ObjectSpanwMetadata
{
public string objectType;
public int objectVariation;
}
    public class DiscretePointClickAgentController : MonoBehaviour
    {
        [SerializeField] private float HandMoveMagnitude = 0.1f;
        public PhysicsRemoteFPSAgentController PhysicsController = null;
        private GameObject InputMode_Text = null;
        private ObjectHighlightController highlightController = null;
        private GameObject throwForceBar = null;
        private bool handMode = false;
        private bool visibleObject = true;
        private bool hidingPhase = false;
        public string onlyPickableObjectId = null;
        public bool disableCollistionWithPickupObject = false;
        private bool continuousMove = true;
        private float continuousMoveTimeSeconds = 0.1f;
        private string clickAction = "PickupObject";
        private int actionThrottleCount = 3; 
        private Dictionary<string, int> actionThrottleDict = new Dictionary<string, int>();
        private string[] throttledActions = new string[] {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"};
        private string lastAction = "";
        private bool lastActionSuccess = true;
        private bool throttleTriggered = false;
        void Start() 
        {
            var Debug_Canvas = GameObject.Find("DebugCanvasPhysics");
            PhysicsController = gameObject.GetComponent<PhysicsRemoteFPSAgentController>();

            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
            Debug_Canvas.GetComponent<Canvas>().enabled = true; 

            highlightController = new ObjectHighlightController(PhysicsController, PhysicsController.maxVisibleDistance, false, 0, 0, true, null, this.clickAction);
            highlightController.SetDisplayTargetText(false);
           
            var crosshair = GameObject.Find("DebugCanvasPhysics/Crosshair");
            if (crosshair) {
                crosshair.SetActive(false);
            }

             
            resetThrottleCounts();
            // actionThrottleDict

            // SpawnObjectToHide("{\"objectType\": \"Plunger\", \"objectVariation\": 1}");
        }

        public void OnEnable() {
            InputMode_Text = GameObject.Find("DebugCanvasPhysics/InputModeText");
            throwForceBar = GameObject.Find("DebugCanvasPhysics/ThrowForceBar");
            var camera =  GetComponentInChildren<Camera>();
            camera.fieldOfView = 90.0f;
            // camera.transform.rotation = Quaternion.Euler(30, 0, 0);
             camera.transform.Rotate(30, 0, 0);
            if (InputMode_Text) {
                InputMode_Text.GetComponent<Text>().text = "Point and Click Mode";
            }
            if (throwForceBar) {
                throwForceBar.SetActive(false);
            }
            
            // InputFieldObj = GameObject.Find("DebugCanvasPhysics/InputField");
            // TODO: move debug input field script from, Input Field and disable here
        }

         public void OnDisable() {
             if (throwForceBar) {
                throwForceBar.SetActive(true);
            }
             // TODO: move debug input field script from, Input Field and enable here
        }

        public void SpawnObjectToHide(string objectMeta) {

            var objectData = new ObjectSpanwMetadata();
            Debug.Log(objectMeta);
		    JsonUtility.FromJsonOverwrite(objectMeta, objectData);
              ServerAction action = new ServerAction(){
                  action = "CreateObject",
                  objectType = objectData.objectType,
                  objectVariation = objectData.objectVariation
              };
            PhysicsController.ProcessControlCommand(action);
            onlyPickableObjectId = objectData.objectType + "|" + objectData.objectVariation;
            this.highlightController.SetOnlyPickableId(onlyPickableObjectId);
            // DisableObjectCollisionWithAgent(onlyPickableObjectId);
        }

         public void SetOnlyPickableObject(string objectId) {
             onlyPickableObjectId = objectId;
            this.highlightController.SetOnlyPickableId(objectId);
        }

        public void SetContinuousMove(int continuous) {
            this.continuousMove = continuous != 0;
        }

        public void SetContinuousMoveSpeed(float continuousMoveTimeSeconds = 0.5f) {
            this.continuousMoveTimeSeconds = continuousMoveTimeSeconds;
        }

        public void SetOnlyObjectId(string objectId) {
            onlyPickableObjectId = objectId;
            this.highlightController.SetOnlyPickableId(objectId);
        }

         public void SetOnlyObjectIdSeeker(string objectId) {
             onlyPickableObjectId = objectId;
            this.highlightController.SetOnlyPickableId(objectId, true);
        }

        public void SetClickAction(string action) {
            this.clickAction = action;
            this.highlightController.SetPickupAction(action);
        }

        public void SetActionThrottleCount(int throttleCount) {
            this.actionThrottleCount = throttleCount;
        } 

        public void SpawnAgent(int randomSeed) {
            ServerAction action = new ServerAction(){
                action = "RandomlyMoveAgent",
                randomSeed = randomSeed
            };
            PhysicsController.ProcessControlCommand(action);
        }

        public void QuitGame() {
            Application.Quit();
        }

        public void DisableObjectCollisionWithAgent(string objectId) {
            var physicsSceneManager = FindObjectOfType<PhysicsSceneManager>();
            if (!physicsSceneManager.UniqueIdToSimObjPhysics.ContainsKey(objectId)) {
                return;
            }
            
            SimObjPhysics target = physicsSceneManager.UniqueIdToSimObjPhysics[objectId];
            disableCollistionWithPickupObject = true;
            foreach (Collider c0 in this.GetComponentsInChildren<Collider>()) {
                foreach (Collider c1 in target.GetComponentsInChildren<Collider>()) {
                    Physics.IgnoreCollision(c0, c1);
                }
            }
        }

        public void SetObjectVisible(bool visible) {
            
                                 visibleObject = visible;
                                 // Debug.Log("Calling disply with "+ visibleObject);
                                 var go = PhysicsController.WhatAmIHolding();
                                PhysicsController.UpdateDisplayGameObject( go, visibleObject);
                                var layer = go.layer;
                                if (!visibleObject) {
                                    // go.layer = LayerMask.NameToLayer("SimObjInvisible");
                                    SetLayerRecursively(go, LayerMask.NameToLayer("SimObjInvisible"));
                                }
                                else {
                                    SetLayerRecursively(go, LayerMask.NameToLayer("SimObjVisible"));
                                }
        }


        // public void TeleportAgent(string actionStr) {
        //     var command = new ServerAction();
		//     JsonUtility.FromJsonOverwrite(actionStr, command);
        //     PhysicsController.ProcessControlCommand(command);
        // }

        private bool throttleCheck(string previousActionName, bool previousActionSuccess, string attemptAction) {
            if (actionThrottleCount >= 0) {
                if (!previousActionSuccess && attemptAction == previousActionName ) {
                    throttleTriggered = true;
                    if (actionThrottleDict[attemptAction] == 0) {
                        
                         //Debug.Log("--Throttled " + attemptAction + " count decrease " + actionThrottleDict[attemptAction]);
                        return false;
                    }
                    else {
                        //Debug.Log("Fail " + attemptAction + " count decrease " + actionThrottleDict[attemptAction]);
                        actionThrottleDict[attemptAction] -= 1;
                    }
                }
                else if (!previousActionSuccess && attemptAction != previousActionName ) {
                    //Debug.Log("--Reset " + attemptAction + " count " + actionThrottleDict[attemptAction]);
                    resetThrottleCounts();
                }
            }
            return true;
        }

        private bool processThrottledCommand(ServerAction command) {
            var doAction = throttleCheck(lastAction, lastActionSuccess, command.action);
            if (doAction) {
                PhysicsController.ProcessControlCommand(command);
            }
            return doAction;
        }

        private void processCommand(ServerAction command) {
            if (throttleTriggered) {
                resetThrottleCounts();
            }
            PhysicsController.ProcessControlCommand(command);
        }

        void Update()
        {
                lastAction = PhysicsController.GetLastAction(out lastActionSuccess);
                // if (!prevLastSuccess && lastActionSuccess && actionThrottleDict.ContainsKey(prevLastAction) &&)

                highlightController.UpdateHighlightedObject(Input.mousePosition);
                highlightController.MouseControls();

                if (PhysicsController.actionComplete) {
                        handMode = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
                        float FlyMagnitude = 1.0f;
                        float WalkMagnitude = 0.25f;
                        if (!handMode && !hidingPhase) {
                            //Func<KeyCode, bool> keyPressedFunc = (x) => Input.GetKey(x);
                            var keyPressedFunc = this.continuousMove ? (Func<KeyCode, bool>)((x) => Input.GetKey(x)) : (Func<KeyCode, bool>)((x) => Input.GetKeyDown(x)); 
                            if(keyPressedFunc(KeyCode.W))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyAhead";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }
                                else
                                {
                                    action.action = "MoveAhead";
                                    action.timeSeconds = this.continuousMoveTimeSeconds;
                                    action.continuous = this.continuousMove;
                                    action.moveMagnitude = WalkMagnitude;
                                    if (PhysicsController.actionComplete) {
                                        processThrottledCommand(action);
                                    }
                                }
                            }

                            if(keyPressedFunc(KeyCode.S))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyBack";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveBack";
                                    action.timeSeconds = this.continuousMoveTimeSeconds;
                                    action.continuous = this.continuousMove;
                                    action.moveMagnitude = WalkMagnitude;		
                                    if (PhysicsController.actionComplete) {
                                        processThrottledCommand(action);
                                    }
                                }
                            }

                            if(keyPressedFunc(KeyCode.A))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyLeft";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveLeft";
                                    action.timeSeconds = this.continuousMoveTimeSeconds;
                                    action.continuous = this.continuousMove;
                                    action.moveMagnitude = WalkMagnitude;		
                                    if (PhysicsController.actionComplete) {
                                        processThrottledCommand(action);
                                    }
                                }
                            }

                            if(keyPressedFunc(KeyCode.D))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyRight";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveRight";
                                    action.timeSeconds = this.continuousMoveTimeSeconds;
                                    action.continuous = this.continuousMove;
                                    action.moveMagnitude = WalkMagnitude;		
                                    if (PhysicsController.actionComplete) {
                                        processThrottledCommand(action);
                                    }
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.I))
                            {
                                if(PhysicsController.FlightMode)
                                {
                                    ServerAction action = new ServerAction();
                                    action.action = "FlyUp";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.K))
                            {
                                if(PhysicsController.FlightMode)
                                {
                                    ServerAction action = new ServerAction();
                                    action.action = "FlyDown";
                                    action.moveMagnitude = FlyMagnitude;
                                    processCommand(action);
                                }
                            }

                            // if(Input.GetKeyDown(KeyCode.UpArrow))
                            // {
                            //     ServerAction action = new ServerAction();
                            //     action.action = "LookUp";
                            //     processCommand(action); 
                            // }

                            // if(Input.GetKeyDown(KeyCode.DownArrow))
                            // {
                            //     ServerAction action = new ServerAction();
                            //     action.action = "LookDown";
                            //     processCommand(action); 
                            // }

                            if(Input.GetKeyDown(KeyCode.LeftArrow) )//|| Input.GetKeyDown(KeyCode.J))
                            {
                                ServerAction action = new ServerAction();
                                // action.action = "RotateLeft";
                                Debug.Log("RotateLeftSmooth process");	
                                action.action = "RotateLeftSmooth";
                                action.timeStep = 0.4f;
                                if (PhysicsController.actionComplete) {
                                    processCommand(action); 
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.RightArrow) )//|| Input.GetKeyDown(KeyCode.L))
                            {
                                ServerAction action = new ServerAction();
                                // action.action = "RotateRight";
                                action.action = "RotateRightSmooth";
                                action.timeStep = 0.4f;
                                //processCommand(action); 

                                 if (PhysicsController.actionComplete) {
                                    processCommand(action); 
                                }
                            }
                        }

                        

                        if (this.PhysicsController.WhatAmIHolding() != null  && handMode)
                        {
                            var actionName = "MoveHandForce";
                            var localPos = new Vector3(0, 0, 0);
                            // Debug.Log(" Key down shift ? " + Input.GetKey(KeyCode.LeftAlt) + " up " + Input.GetKeyDown(KeyCode.UpArrow));
                            if (Input.GetKeyDown(KeyCode.W)) {
                                localPos.y += HandMoveMagnitude;
                            }
                            else if (Input.GetKeyDown(KeyCode.S)) {
                                localPos.y -= HandMoveMagnitude;
                            }
                            else if (Input.GetKeyDown(KeyCode.UpArrow)) {
                                localPos.z += HandMoveMagnitude;
                            }
                            else if (Input.GetKeyDown(KeyCode.DownArrow)) {
                                localPos.z -= HandMoveMagnitude;
                            }
                            else if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A)) {
                                localPos.x -= HandMoveMagnitude;
                            }
                            else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D)) {
                                localPos.x += HandMoveMagnitude;
                            }
                            if (actionName != "" && localPos.sqrMagnitude > 0) {
                                ServerAction action = new ServerAction
                                {
                                    action = "MoveHandForce",
                                    x = localPos.x,
                                    y = localPos.y,
                                    z = localPos.z
                                };
                                this.processCommand(action);
                            }

                            if (Input.GetKeyDown(KeyCode.Space)) {
                                SetObjectVisible(true);
                                  var action = new ServerAction
                                    {
                                        action = "DropHandObject",
                                        forceAction = true
                                    };
                                this.processCommand(action);
                            }
                        }
                        else if (handMode) {
                            if (Input.GetKeyDown(KeyCode.Space)) {
                                var withinReach = PhysicsController.FindObjectInVisibleSimObjPhysics(onlyPickableObjectId) != null;
                                if (withinReach) {
                                    ServerAction action = new ServerAction();
                                    action.objectId = onlyPickableObjectId;
                                    action.action = this.clickAction;
                                    processCommand(action);
                                }
                            }
                        }
                        if ((Input.GetKeyDown(KeyCode.LeftControl) || Input.GetKeyDown(KeyCode.C) || Input.GetKeyDown(KeyCode.RightControl) ) && PhysicsController.actionComplete) {
                            ServerAction action = new ServerAction();
                            if (this.PhysicsController.isStanding()) {
                                action.action = "Crouch";
                                processCommand(action);
                            }
                            else {
                                 action.action = "Stand";
                                 processCommand(action);
                            }
                            

                        }

                        if (PhysicsController.WhatAmIHolding() != null) {
                             if (Input.GetKeyDown(KeyCode.Space) && !hidingPhase && !handMode) {
                                
                                SetObjectVisible(!visibleObject);
                                //  visibleObject = !visibleObject;
                                //  // Debug.Log("Calling disply with "+ visibleObject);
                                //  var go = PhysicsController.WhatAmIHolding();
                                // PhysicsController.UpdateDisplayGameObject( go, visibleObject);
                                // var layer = go.layer;
                                // if (!visibleObject) {
                                //     // go.layer = LayerMask.NameToLayer("SimObjInvisible");
                                //     SetLayerRecursively(go, LayerMask.NameToLayer("SimObjInvisible"));
                                // }
                                // else {
                                //     SetLayerRecursively(go, LayerMask.NameToLayer("SimObjVisible"));
                                // }
                                // Debug.Log("prev layer " + layer + " new layer " + go.layer);
                                // var go = PhysicsController.WhatAmIHolding();
                                //  foreach (MeshRenderer mr in go.GetComponentsInChildren<MeshRenderer>() as MeshRenderer[]) {
                                //      mr.enabled = !visibleObject;
                                //  }
                             }
                            //   if (Input.GetKeyDown(KeyCode.H)) { 
                            //       // HIDING PHASE
                            //       this.hidingPhase = true;
                            //       visibleObject = true;
                            //       PhysicsController.UpdateDisplayGameObject( PhysicsController.WhatAmIHolding(), true);
                         
                            //   }
                        }
            }
        }

        private void resetThrottleCounts(string action = "") {
            throttleTriggered = false;
            if (action == "") {
                foreach (var ac in throttledActions) {
                    actionThrottleDict[ac] = actionThrottleCount;
                }
            }
            else {
                actionThrottleDict[action] = actionThrottleCount;
            }
        }

        private static void SetLayerRecursively(GameObject obj, int newLayer)
    {
        if (null == obj)
        {
            return;
        }
       
        obj.layer = newLayer;
       
        foreach (Transform child in obj.transform)
        {
            if (null == child)
            {
                continue;
            }
            SetLayerRecursively(child.gameObject, newLayer);
        }
    }
    }
    
}