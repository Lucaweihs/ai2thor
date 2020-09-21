console.log("GAME URL: ", window.game_url);
let gameInstance = null;
let getParams = parseGet();
let hider = getParams['role'] !== 'seeker';
let gameInitialized  = false;
let objectId = '';
let spawnRandomSeed = 'spawnSeed' in getParams ? parseInt(getParams['spawnSeed']) : 0;
let objectsRandomSeed = 'objectsSeed' in getParams ? parseInt(getParams['objectsSeed']) : 0;
let outputData = {
  object_type: getParams['object'],
  object_variation: getParams['variation'],
  open_objects: []
};
let gameConfig = null;
let lastMetadadta = null;
let reachablePositions = null;

$(
  () => {
    // Async loading disabled due to possible race conditions
    //  if (getParams['player'] === 'seeker') {
    //    $.getJSON(getParams['config'], function(json) {
    //       gameConfig = json;
    //    });
    // }
    $('#role-str').html(('role' in getParams ? getParams['role'] : 'hider').toUpperCase());
    if (hider) {
        let objectHtml = `<strong class="important-text">${getParams['object']}</strong>`;
        $("#instruction-text").html(`You have to hide a ${objectHtml}`);
        $("#instruction-2").html(`Move around in the room, open drawers and cabinets to look for a good hiding spot.`);
        $("#instruction-3").html('When you are ready, move the object (see Shift controls) to place it more precisely, click on it to drop it.');
        $("#instruction-4").html(`If you're happy with your hiding spot click the <strong class="green-text">Finish</strong> button above. Or <strong class="red-text">Reset</strong> to start over.`);
         $("#instruction-5").html("If a door/drawer opens and closes, it means you're in the way! Move back and try again.");
        $("#instructions-hider").show();
      initGame(window.game_url);
    }
    else {
      $.getJSON(
          `https://thor-turk.s3-us-west-2.amazonaws.com/hide-n-seek/data/hider/${getParams['config']}`,
          function(config) {
              console.log("------ Config", config);
              gameConfig = config;
              $("#instructions-seeker").show();
              $("#message-text").show();
              // gameConfig['agentPosition']
              let objectHtml = `<strong class="important-text">${gameConfig.object_type}</strong>`;
              $("#instruction-text").html(`You have to find a ${objectHtml}`);
              $("#instruction-2").html(`Move around in the room, open drawers and cabinets to look for a ${objectHtml}.`);
              $("#instruction-3").html('Click on it once you have found it.');
              $("#instruction-4").html(`If you cannot find the ${objectHtml} after some time, you can click the <strong class="red-text">Give Up</strong> button above.`);
              getParams['object'] = gameConfig['object_type'];
              getParams['variation'] = gameConfig['object_variation'];

              outputData.object_type = gameConfig['object_type'];
              outputData.object_variation = gameConfig['object_variation'];

              $("#finish-hit").click((e) => {

                    outputData['success'] = false;
                    submitHit(lastMetadadta);

              }).text("Give Up").toggleClass("giveup-btn");


              $("#reset-hit").hide();

              setTimeout(() => {
                   $("#finish-hit").attr("disabled", false);
              }, 'giveUpEnableSeconds' in getParams ? parseInt(getParams['giveUpEnableSeconds']) * 1000 : 30000);

              initGame(window.game_url);
       });
    }
  }
);

window.game_init = function() {
  if (!gameInitialized) {
    gameInitialized = true;
  }
};

window.onUnityMetadata = function(metadata) {

   let jsonMeta = JSON.parse(metadata);

  // FIRST init event
   if (jsonMeta.agents[0].lastAction === null) {
      console.log("--- ", jsonMeta.agents[0].sceneName)
      outputData['scene'] = jsonMeta.agents[0].sceneName;
      outputData['trayectory'] = [];
      outputData['actions'] = [];

       gameInstance.SendMessage ('FPSController', 'Step', JSON.stringify({
                    action: "RandomizeHideSeekObjects",
                    randomSeed: objectsRandomSeed,
                    removeProb: 0.0,
       }));

        ["Cup", "Mug", "Bread", "Tomato", "Plunger", "Knife"]
            .forEach(
                k => gameInstance.SendMessage(
                    'FPSController',
                    'Step',
                    JSON.stringify({action: "DisableAllObjectsOfType", objectId: k})
                )
            );

        if (hider) {
            gameInstance.SendMessage('FPSController', 'SpawnAgent', spawnRandomSeed);
        }
        else {
            gameInstance.SendMessage('FPSController', 'Step', JSON.stringify({
                action: "TeleportFull",
                x: gameConfig.agent_start_location.x,
                y: gameConfig.agent_start_location.y,
                z: gameConfig.agent_start_location.z,
                horizon:  gameConfig.agent_start_location.horizon,
                rotation: {x: 0.0, y: gameConfig.agent_start_location.rotation, z: 0.0},
                standing: gameConfig.agent_start_location.standing
            }));
            gameInstance.SendMessage('FPSController', 'Step', JSON.stringify({
                action: "CreateObjectAtLocation",
                position: gameConfig.object_position,
                rotation: gameConfig.object_rotation,
                forceAction: true,
                objectType: gameConfig.object_type,
                objectVariation: gameConfig.object_variation,
                randomizeObjectAppearance: false
            }));
        }
   }
   else {
     if (jsonMeta.agents[0].lastAction === "CreateObject" && !jsonMeta.agents[0].lastActionSuccess) {
        throw `Action '${jsonMeta.agents[0].lastAction}' failed with error: "${jsonMeta.agents[0].errorMessage}"' `
     }
   }
    let agentMetadata = jsonMeta.agents[0];
    let agent = agentMetadata.agent;
    outputData.actions.push({
        lastAction: agentMetadata.lastAction,
        lastActionSuccess: agentMetadata.lastActionSuccess,
        agent: {
           x: agent.position.x,
           y: agent.position.y,
           z: agent.position.z,
           rotation: agent.rotation.y,
           horizon: agent.cameraHorizon,
           standing: agentMetadata.isStanding
        }
    });

      handleEvent(jsonMeta);
   lastMetadadta = jsonMeta;
};

window.onUnityEvent = function(event) {
  let jsonEvt = JSON.parse(event);
};

function gatherFinalState(metadata) {
  let agentMetadata = metadata.agents[0];
  let filtered = agentMetadata.objects.filter((obj) => obj.objectId === objectId);
  if (filtered.length === 1) {
    let object = filtered[0];

    outputData['object_position'] = object.position;
    outputData['object_rotation'] = object.rotation;
    outputData['open_objects'] = agentMetadata.objects.filter((obj) => obj.isOpen).map(obj => obj.objectId);

     outputData['object_locations_and_rotations'] = agentMetadata.objects.reduce((acc, obj, {}) => {
          return {
              ...acc,
               [obj.objectId]:{
                    position: obj.position,
                    rotation: obj.rotation
               }
          }
      });
  }
  else {
    throw `Invalid id ${objectId} in metadata.objects: ${agentMetadata.objects}`;
  }
  return outputData;
}

 function submitHit(metadata) {
    let data = gatherFinalState(metadata);
    document.forms['mturk_form'].assignmentId.value = getParams['assignmentId'];
    console.log('Turk submit!!', data);
    document.forms['mturk_form'].data.value = JSON.stringify(data);
    document.forms['mturk_form'].submit();
}

function debugPrintJson(metadata) {
    let data = gatherFinalState(metadata);
    console.log('Output Json:', data);
}

function Move(metadata) {
  outputData.trayectory.push(metadata.agents[0].agent.position);
}

function DropObject(metadata) {
   $("#finish-hit").attr("disabled", false);
}

function CreateObject(metadata) {
  let agentMetadata = metadata.agents[0];
  objectId = agentMetadata.actionReturn;
  outputData['target_id'] = objectId;

  $("#finish-hit").click((e) => {
      // The callback for ExhaustiveSearchForItem action will call submitHit
      // with visibility information of the hiding spot
      gameInstance.SendMessage ('FPSController', 'Step', JSON.stringify({
        "action": "ExhaustiveSearchForItem",
        "objectId": objectId,
        "positions": reachablePositions
    }));
  });

  $("#debug-json").click((e) => {
    debugPrintJson(lastMetadadta)
  });

  $("#reset-hit").click((e) => {
  	console.log("-----------RESET HIT!")
  	$("#reset-hit").blur();
    gameInstance.SendMessage ('PhysicsSceneManager', 'SwitchScene', outputData.scene);

  }).attr("disabled", false);

  $("#move").click((e) => {
       gameInstance.SendMessage ('FPSController', 'Step', JSON.stringify({
          action: "MoveAhead",
          moveMagnitude:  0.25
      }));
  })

}

function OpenObject(metadata) {
  let agentMetadata = metadata.agents[0];
}

function FoundObject(metadata) {
  let agentMetadata = metadata.agents[0];
  let pickedObjectId = agentMetadata.actionReturn;
    if (pickedObjectId === objectId) {
      console.log("Success!!");
    outputData['success'] = true;
    $("#message-text").html("<strong class='green-text'>You Found the object!</strong>").show();
    // Auto submit
    setTimeout(() => {
        submitHit(lastMetadadta)
    }, 1000);
  }
}

function RegisterAgentPosition(metadata) {
  let agentMetadata = metadata.agents[0];
  let agent = agentMetadata.agent;
  outputData['agent_start_location'] =
    {
        "x": agent.position.x,
        "y": agent.position.y,
        "z": agent.position.z,
        "rotation": agent.rotation.y,
        "horizon": agent.cameraHorizon,
        "standing": agentMetadata.isStanding
    };
   let objectName = getParams['object'];
   let objectVariation = parseInt(getParams['variation']);
   c = agentMetadata.actionReturn;
    gameInstance.SendMessage('FPSController', 'SpawnObjectToHide',  JSON.stringify(
            {
                      objectType: objectName,
                      objectVariation: objectVariation,
            }
      ));
}

function CreateObjectAtLocation(metadata) {
    let agentMetadata = metadata.agents[0];
    gameInstance.SendMessage('FPSController', 'SetOnlyObjectIdSeeker', agentMetadata.actionReturn);
    objectId = gameConfig.target_id;
}

function ExhaustiveSearchForItem(metadata) {
        let agentMetadata = metadata.agents[0];
    outputData['visibility'] = {
        objectSeen: agentMetadata.actionReturn['objectSeen'],
        positionsTried: agentMetadata.actionReturn['positionsTried']
    };
    submitHit(metadata);
}

let eventHandlers = {
    hider: {
      MoveAhead: Move,
      MoveBack: Move,
      MoveLeft: Move,
      MoveRight: Move,
      DropHandObject: DropObject,
      CreateObject: CreateObject,
      OpenObject: OpenObject,
      RandomlyMoveAgent: RegisterAgentPosition,
      ExhaustiveSearchForItem: ExhaustiveSearchForItem
    },
    seeker: {
      MoveAhead: Move,
      MoveBack: Move,
      MoveLeft: Move,
      MoveRight: Move,
      OpenObject: OpenObject,
      PickupObject: FoundObject,
      CreateObjectAtLocation: CreateObjectAtLocation
    }
};

function handleEvent(metadata) {
  let action = metadata.agents[0].lastAction;
  let role = hider ? 'hider' : 'seeker';
  let handler = eventHandlers[role][action];
  if (handler !== undefined) {
        handler(metadata);
  }
}

function initGame(url) {
  const t0 = performance.now();
  gameInstance = UnityLoader.instantiate("gameContainer", url, {
    onProgress: UnityProgress, Module: {
      onRuntimeInitialized: function () {
        UnityProgress(gameInstance, "complete");
        const t1 = performance.now();
        console.log(`Load finished. Took: ${(t1 - t0) / 1000}s`);
        var container = document.getElementById("gameContainer");
      },
    }
  });
}

function paramStrToAssocArray(prmstr) {
  let params = {};
  let prmarr = prmstr.split('&');
  for (let i = 0; i < prmarr.length; i++) {
    let tmparr = prmarr[i].split('=');
    params[tmparr[0]] = tmparr[1];
  }
  return params;
}

function parseGet() {
  let paramStr = window.location.search.substr(1);
  return paramStr !== null && paramStr !== ''
    ? paramStrToAssocArray(paramStr)
    : {};
}