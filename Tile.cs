using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Tile : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    void OnTriggerExit(Collider obj)
    {
        if (obj.tag == "Player")
        {
            TilesManager.Instance.SpawnTile();
            StartCoroutine(FallDown());
        }
    }
    IEnumerator FallDown()
    {
        yield return new WaitForSeconds(1);
        GetComponent<Rigidbody>().isKinematic = false;
        yield return new WaitForSeconds(1);
        GetComponent<Rigidbody>().isKinematic = true;
        if(name == "TopTile")
        {
            TilesManager.Instance.AddTopTile(gameObject);

        }
        else
        {
            TilesManager.Instance.AddLeftTile(gameObject);
        }
    }
}
