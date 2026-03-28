# Streamlit Community Cloud Draft Deployment

This app can be deployed as a draft and updated later by pushing new changes to the same GitHub repository.

## App entrypoint

- Main file: `02_app/app.py`

## Required secrets

Add these in Streamlit Community Cloud under the app's **Advanced settings** or **Secrets**:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
EIA_API_KEY = "your_eia_api_key"
```

Optional values:

```toml
SODIR_API_BASE_URL = "https://factpages.sodir.no/en/wellbore/tableview/exploration/all"
APP_ENV = "draft"
CACHE_TTL_SECONDS = "3600"
```

## Deployment steps

1. Create a GitHub repository for this folder.
2. Push the project to GitHub.
3. Sign in to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Set the main file path to `02_app/app.py`.
6. Paste the required secrets into the app settings.
7. Deploy.

## Draft workflow

- Keep the app in draft form by continuing to edit the same repository.
- Push changes to GitHub whenever you want the deployed app updated.
- Streamlit Community Cloud will rebuild from the latest commit.
