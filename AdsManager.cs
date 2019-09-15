using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using GoogleMobileAds.Api;
public class AdsManager : MonoBehaviour
{

    string appid = "ca-app-pub-2048206714429898~3916497460";
    string gameoverid = "ca-app-pub-2048206714429898/1519588357";
    string bottomadid = "ca-app-pub-2048206714429898/7001825807";
    private static AdsManager instance;
    private BannerView bannerView;
    private InterstitialAd interstitialAd;

    public static AdsManager Instance
    {
        get
        {
            if(instance == null)
            {
                instance = GameObject.FindObjectOfType<AdsManager>();
            }
            return instance;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        MobileAds.Initialize(appid);
        bannerView = new BannerView(bottomadid, AdSize.Banner, AdPosition.Bottom);
        interstitialAd = new InterstitialAd(gameoverid);
        AdRequest bannerRequest = new AdRequest.Builder().Build();
        bannerView.LoadAd(bannerRequest);
        bannerView.OnAdLoaded += BannerView_OnAdLoaded;
        AdRequest intersitialRequest = new AdRequest.Builder().Build();
        interstitialAd.LoadAd(intersitialRequest);
        interstitialAd.OnAdClosed += InterstitialAd_OnAdClosed;

    }
    private void InterstitialAd_OnAdClosed(object sender, System.EventArgs e)
    {
        bannerView.Show();
    }
    private void BannerView_OnAdLoaded(object sender,System.EventArgs e)
    {
        bannerView.Show();
    }
    // Update is called once per frame
    void Update()
    {
        
    }
    public void ShowGameOverAd()
    {
        bannerView.Hide();
        if (interstitialAd.IsLoaded())
        {
            interstitialAd.Show();
        }
    }
}
