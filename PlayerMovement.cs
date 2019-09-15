using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class PlayerMovement : MonoBehaviour
{
    public int speed;
    private Vector3 dir;
    public GameObject ps;
    public Text ScoreText;
    private int score=0;
    private bool isDead;
    public Color BackgroundColor;
    
  
    
    public Image BackgroundImage;
    public Text[] scoreTexts;
    public GameObject BestScoreText;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0)&& !isDead)
        {
            if(dir == Vector3.forward)
            {
                dir = Vector3.left;
            }
            else
            {
                dir = Vector3.forward;
            }

            score += 1;
            ScoreText.text = "Score: " + score.ToString();

        }

        
        float amountToMove = speed * Time.deltaTime;
        transform.Translate(dir * amountToMove);
        
    }
    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag == "Pickup")
        {
            score += 3;
            ScoreText.text = "Score: " + score.ToString();
            other.gameObject.SetActive(false);
            Instantiate(ps, transform.position, Quaternion.identity);
        }
    }
    private void OnTriggerExit(Collider other)
    {
        if (other.tag == "Tile")
        {
            RaycastHit hit;
            Ray downRay = new Ray(transform.position, -Vector3.up);
            if(!Physics.Raycast(downRay,out hit))
            {
                isDead = true;
                transform.GetChild(0).transform.SetParent(null);
                AdsManager.Instance.ShowGameOverAd();
                ShowScores();
                TilesManager.Instance.Dead();
                Destroy(gameObject);
            }
        }
    }
    void ShowScores()
    {
        ScoreText.gameObject.SetActive(false);
        scoreTexts[1].text = score.ToString();
        int bestScore = PlayerPrefs.GetInt("BestScore", 0);
        if (score > bestScore)
        {
            BackgroundImage.color = BackgroundColor;
            foreach (var item in scoreTexts)
            {
                item.color = new Color(255, 255, 255, 255);
            }
            BestScoreText.SetActive(true);
            bestScore = score;
            PlayerPrefs.SetInt("BestScore", bestScore);
        }
        scoreTexts[3].text = bestScore.ToString();
    }

}
