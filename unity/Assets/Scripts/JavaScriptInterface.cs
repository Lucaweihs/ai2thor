using UnityEngine;
using System.Runtime.InteropServices;


public class JavaScriptInterface : MonoBehaviour {

    [DllImport("__Internal")]
    private static extern void Init();

    [DllImport("__Internal")]
    private static extern void SendEvent(string str);

    [DllImport("__Internal")]
    private static extern void SendMetadata(string str);

    public void SendAction(ServerAction action)
    {
        SendEvent(JsonUtility.ToJson(action));
    }

/*
    metadata: serialized metadata, commonly an instance of MultiAgentMetadata
 */
    public void SendActionMetadata(string metadata)
    {
        SendMetadata(metadata);
    }

    void Start()
    {
        Init();

        Debug.Log("Calling store data");
    }
}
