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
        // private string onlyPickableObjectId = null;
        void Start() 
        {
            var Debug_Canvas = GameObject.Find("DebugCanvasPhysics");
            PhysicsController = gameObject.GetComponent<PhysicsRemoteFPSAgentController>();

            Cursor.visible = true;
            Cursor.lockState = CursorLockMode.None;
            Debug_Canvas.GetComponent<Canvas>().enabled = true; 

            highlightController = new ObjectHighlightController(PhysicsController, PhysicsController.maxVisibleDistance, false, 0, 0, true);
            highlightController.SetDisplayTargetText(false);

            SpawnObjectToHide("{\"objectType\": \"Plunger\", \"objectVariation\": 1}");
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
            this.highlightController.SetOnlyPickableId(objectData.objectType + "|" + objectData.objectVariation);
        }

         public void SetOnlyPickableObject(string objectId) {
            this.highlightController.SetOnlyPickableId(objectId);
        }

        public void SetOnlyObjectId(string objectId) {
            this.highlightController.SetOnlyPickableId(objectId);
        }

         public void SetOnlyObjectIdSeeker(string objectId) {
            this.highlightController.SetOnlyPickableId(objectId, true);
        }

        public void SpawnAgent(int randomSeed) {
            ServerAction action = new ServerAction(){
                action = "RandomlyMoveAgent",
                randomSeed = randomSeed
            };
            PhysicsController.ProcessControlCommand(action);
        }

        public void Step(string serverAction)
		{
			ServerAction controlCommand = new ServerAction();
			JsonUtility.FromJsonOverwrite(serverAction, controlCommand);
			PhysicsController.ProcessControlCommand(controlCommand);
		}

        // public void TeleportAgent(string actionStr) {
        //     var command = new ServerAction();
		//     JsonUtility.FromJsonOverwrite(actionStr, command);
        //     PhysicsController.ProcessControlCommand(command);
        // }

        void Update()
        {
                highlightController.UpdateHighlightedObject(Input.mousePosition);
                highlightController.MouseControls();

                if (PhysicsController.actionComplete) {
                        float FlyMagnitude = 1.0f;
                        float WalkMagnitude = 0.25f;
                        if (!handMode && !hidingPhase) {
                            if(Input.GetKeyDown(KeyCode.W))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyAhead";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveAhead";
                                    action.moveMagnitude = WalkMagnitude;		
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.S))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyBack";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveBack";
                                    action.moveMagnitude = WalkMagnitude;		
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.A))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyLeft";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveLeft";
                                    action.moveMagnitude = WalkMagnitude;		
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.D))
                            {
                                ServerAction action = new ServerAction();
                                if(PhysicsController.FlightMode)
                                {
                                    action.action = "FlyRight";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }

                                else
                                {
                                    action.action = "MoveRight";
                                    action.moveMagnitude = WalkMagnitude;		
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.I))
                            {
                                if(PhysicsController.FlightMode)
                                {
                                    ServerAction action = new ServerAction();
                                    action.action = "FlyUp";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            if(Input.GetKeyDown(KeyCode.K))
                            {
                                if(PhysicsController.FlightMode)
                                {
                                    ServerAction action = new ServerAction();
                                    action.action = "FlyDown";
                                    action.moveMagnitude = FlyMagnitude;
                                    PhysicsController.ProcessControlCommand(action);
                                }
                            }

                            // if(Input.GetKeyDown(KeyCode.UpArrow))
                            // {
                            //     ServerAction action = new ServerAction();
                            //     action.action = "LookUp";
                            //     PhysicsController.ProcessControlCommand(action); 
                            // }

                            // if(Input.GetKeyDown(KeyCode.DownArrow))
                            // {
                            //     ServerAction action = new ServerAction();
                            //     action.action = "LookDown";
                            //     PhysicsController.ProcessControlCommand(action); 
                            // }

                            if(Input.GetKeyDown(KeyCode.LeftArrow) )//|| Input.GetKeyDown(KeyCode.J))
                            {
                                ServerAction action = new ServerAction();
                                // action.action = "RotateLeft";
                                action.action = "RotateLeftSmooth";
                                action.timeStep = 0.4f;
                                PhysicsController.ProcessControlCommand(action); 
                            }

                            if(Input.GetKeyDown(KeyCode.RightArrow) )//|| Input.GetKeyDown(KeyCode.L))
                            {
                                ServerAction action = new ServerAction();
                                // action.action = "RotateRight";
                                action.action = "RotateRightSmooth";
                                action.timeStep = 0.4f;
                                PhysicsController.ProcessControlCommand(action); 
                            }
                        }

                         if (Input.GetKeyDown(KeyCode.LeftShift) || Input.GetKeyDown(KeyCode.RightShift)) {
                            handMode = true;
                         }
                         if (Input.GetKeyUp(KeyCode.LeftShift) || Input.GetKeyUp(KeyCode.RightShift)){
                            handMode = false;
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
                                this.PhysicsController.ProcessControlCommand(action);
                            }

                            if (Input.GetKeyDown(KeyCode.Space)) {
                                  var action = new ServerAction
                                    {
                                        action = "DropHandObject",
                                        forceAction = true
                                    };
                                this.PhysicsController.ProcessControlCommand(action);
                            }
                        }
                        if (Input.GetKeyDown(KeyCode.LeftControl) || Input.GetKeyDown(KeyCode.C) || Input.GetKeyDown(KeyCode.RightControl) ) {
                            ServerAction action = new ServerAction();
                            if (this.PhysicsController.isStanding()) {
                                action.action = "Crouch";
                                PhysicsController.ProcessControlCommand(action);
                            }
                            else {
                                 action.action = "Stand";
                            }
                            PhysicsController.ProcessControlCommand(action);

                        }

                        if (PhysicsController.WhatAmIHolding() != null) {
                             if (Input.GetKeyDown(KeyCode.Space) && !hidingPhase) {
                                
                                 visibleObject = !visibleObject;
                                  Debug.Log("Calling disply with "+ visibleObject);
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
                                // Debug.Log("prev layer " + layer + " new layer " + go.layer);
                                // var go = PhysicsController.WhatAmIHolding();
                                //  foreach (MeshRenderer mr in go.GetComponentsInChildren<MeshRenderer>() as MeshRenderer[]) {
                                //      mr.enabled = !visibleObject;
                                //  }
                             }
                              if (Input.GetKeyDown(KeyCode.H)) { 
                                  // HIDING PHASE
                                  this.hidingPhase = true;
                                  visibleObject = true;
                                  PhysicsController.UpdateDisplayGameObject( PhysicsController.WhatAmIHolding(), true);
                         
                              }
                        }
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