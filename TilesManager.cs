using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class TilesManager : MonoBehaviour
{
    
    public GameObject CurrentTile;
    public GameObject[] tilesPrefab;
    public GameObject canvas;
    Stack<GameObject> topTiles = new Stack<GameObject>();
    Stack<GameObject> leftTiles = new Stack<GameObject>();

    private static TilesManager instance;

    public static TilesManager Instance
    {
        get
        {
            if (instance == null)
                instance = GameObject.FindObjectOfType<TilesManager>();
            return instance;

        }
        
    }

    // Start is called before the first frame update
    void Start()
    {
        
        CreateTiles(100);
        
        for (int i = 0; i < 50; i++)
        {


            SpawnTile();
            
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void CreateTiles(int amount)
    {
        for(int i = 0; i < amount; i++)
        {
            leftTiles.Push(Instantiate(tilesPrefab[1]));
            topTiles.Push(Instantiate(tilesPrefab[0]));
            topTiles.Peek().SetActive(false);
            leftTiles.Peek().SetActive(false);
            topTiles.Peek().name = "TopTile";
            leftTiles.Peek().name = "LeftTile";

        }
    }
    public void SpawnTile()
    {
        if (leftTiles.Count==0 || topTiles.Count == 0)
        {
            CreateTiles(10);
        }
        int index = Random.Range(0, 2);
        if (index == 0)
        {
            GameObject tmp = topTiles.Pop();
            tmp.SetActive(true);
            tmp.transform.position= CurrentTile.transform.GetChild(0).GetChild(index).position;
            CurrentTile = tmp;
        }
        else if(index == 1)
        {
            GameObject tmp = leftTiles.Pop();
            tmp.SetActive(true);
            tmp.transform.position= CurrentTile.transform.GetChild(0).GetChild(index).position;
            CurrentTile = tmp;
        }
        int doPickup = Random.Range(0, 10);
        if (doPickup == 0)
        {
            CurrentTile.transform.GetChild(1).gameObject.SetActive(true);
        }
        
    }
    public void AddTopTile(GameObject obj)
    {
        topTiles.Push(obj);
        topTiles.Peek().SetActive(false);
    }
    public void AddLeftTile(GameObject obj)
    {
        leftTiles.Push(obj);
        leftTiles.Peek().SetActive(false);
   
    }
    public void Dead()
    {
        canvas.SetActive(true);
        canvas.transform.GetChild(0).gameObject.GetComponent<Animator>().SetTrigger("GameOver");

    }
    public void RestartGame()
    {
        SceneManager.LoadScene(0);
    }
}
